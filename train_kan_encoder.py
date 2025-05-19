# train.py
import os
import sys
import time
import argparse
import torch
import numpy as np
import torch.amp
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm
from model.model_new import KANCompressor
from dataset import ExternalImageDataset as CompressDataset
from metric_torch import psnr
from ssim import ssim as msssim
import csv


class DynamicRateDistortionLoss(torch.nn.Module):
    def __init__(self, lmbda=0.1, beta=0.05, target_sparsity=0.3):
        super().__init__()
        self.lmbda = lmbda
        self.beta = beta
        self.mse_loss = torch.nn.MSELoss()
        self.base_target_sparsity = target_sparsity

    def forward(self, recon_x, x, loss_dict):
        actual_sparsity = loss_dict['sparsity']
        target_sparsity_tensor = torch.tensor(
            self.base_target_sparsity, 
            device=actual_sparsity.device
        )
        current_step = loss_dict['global_step']
        anneal_factor = min(current_step / 5000, 1.0)
        
        mse = self.mse_loss(recon_x, x)
        ssim = msssim(recon_x, x)
        
        target_sparsity_scalar = min(
            self.base_target_sparsity, 
            self.base_target_sparsity/2 + self.base_target_sparsity * anneal_factor
        ) * self.base_target_sparsity
        target_sparsity = torch.tensor(
            target_sparsity_scalar,
            device=loss_dict['sparsity'].device,
            dtype=loss_dict['sparsity'].dtype
        )
        sparsity_loss = torch.abs(actual_sparsity - target_sparsity_tensor)
        
        return (
            0.7 * (1 - ssim) + 
            0.3 * mse + 
            self.lmbda * loss_dict['commit_loss'] * (1 + 0.5 * anneal_factor) +
            self.beta * sparsity_loss
        )


def create_output_dir(args):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dir_name = f"{timestamp}_sp{args.target_sparsity}_ep{args.epochs}_lr{args.lr}_bs{args.batch_size}"
    output_path = os.path.join("output", dir_name)
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)
    return output_path


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} with config:")
    print(f"-> Sparsity Target: {args.target_sparsity}")
    print(f"-> Batch Size: {args.batch_size}")
    print(f"-> Learning Rate: {args.lr}")
    
    output_dir = create_output_dir(args)
    print(f"All outputs will be saved to: {output_dir}")

    csv_file = "training_metrics.csv"

    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "PSNR (dB)", "MS-SSIM", "Sparsity", "LR"])
    
    dataset = CompressDataset(args.data_dir)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    model = KANCompressor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
        betas=(0.9, 0.98)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    criterion = DynamicRateDistortionLoss(lmbda=0.15, beta=0.1, target_sparsity=args.target_sparsity)
    
    global_step = 0
    best_metrics = {'psnr': 0.0, 'ssim': 0.0}
    
    scaler = torch.amp.GradScaler()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            dynamic_ncols=True
        )
        
        for batch_idx, (x) in enumerate(progress_bar):
            x = x.to(device, non_blocking=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                recon_x,  loss_dict = model(x, target_sparsity=args.target_sparsity)
                loss_dict['global_step'] = global_step
                loss = criterion(recon_x, x, loss_dict)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            current_psnr = psnr(recon_x, x)
            current_ssim = msssim(recon_x, x)
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "PSNR": f"{current_psnr:.4f}",
                "SSIM": f"{current_ssim:.4f}",
                "Sparsity": f"{loss_dict['sparsity']:.3f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            if global_step % 1 == 0:
                save_image(
                    recon_x[:4],
                    os.path.join(output_dir, "images", f"step{global_step}.png"),
                    nrow=2,
                    normalize=True
                )
            
            global_step += 1
        
        scheduler.step()
        
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_psnr": best_metrics['psnr'],
        }
        # torch.save(
        #     checkpoint,
        #     os.path.join(output_dir, "checkpoints", f"epoch_{epoch+1}.pth")
        # )

        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch+1, 
                f"{loss.item():.4f}", 
                f"{current_psnr:.4f}", 
                f"{current_ssim:.4f}", 
                f"{loss_dict['sparsity']:.3f}", 
                f"{optimizer.param_groups[0]['lr']:.2e}"
            ])
        
        if current_psnr > best_metrics['psnr']:
            best_metrics.update({
                "psnr": current_psnr,
                "ssim": current_ssim,
                "epoch": epoch + 1
            })
            # torch.save(
            #     model.state_dict(),
            #     os.path.join(output_dir, "checkpoints", "best_model.pth")
            # )
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Avg Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"Current PSNR/SSIM: {current_psnr:.2f}/{current_ssim:.4f}")
        print(f"Best PSNR/SSIM: {best_metrics['psnr']:.2f}/{best_metrics['ssim']:.4f} (Epoch {best_metrics['epoch']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAN image compression training script")
    parser.add_argument("--data_dir", type=str, default="./data", help="training dataset path")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.15, help="learning rate")
    parser.add_argument("--target_sparsity", type=float, default=0.3, help="target sparsity")
    
    args = parser.parse_args()
    
    print("\n" + "="*40)
    print("training config:")
    for arg in vars(args):
        print(f"{arg:>15}: {getattr(args, arg)}")
    print("="*40 + "\n")
    
    train(args)
