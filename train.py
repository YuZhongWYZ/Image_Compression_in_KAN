import torch
from torch.utils.data import DataLoader
import numpy as np
import csv
import os
import argparse
from datetime import datetime
from dataset import CompressDataset
from model.model import KANCompressor
from metric_torch import  psnr
from ssim import ssim as msssim
from tqdm.auto import tqdm  
import torchvision

class RateDistortionLoss(torch.nn.Module):
    def __init__(self, lambda_rd=5e-2):
        super().__init__()
        self.lambda_rd = lambda_rd

    def forward(self, x_hat, x, logits):
        prob = torch.softmax(logits, dim=1)
        rate = torch.mean(-prob * torch.log2(prob + 1e-8))
        mse_loss = torch.mean((x_hat - x)**2)
        ssim_loss = 1 - msssim(x_hat, x)
        return 0.4*ssim_loss + 0.6*mse_loss + self.lambda_rd * rate

def save_reconstructed_images(x_hat, epoch, batch_idx, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    x_hat = (x_hat + 1) / 2  # [-1,1] -> [0,1]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"epoch{epoch:03d}_batch{batch_idx:03d}_{timestamp}.png"
    torchvision.utils.save_image(x_hat, os.path.join(output_dir, filename))

def train(args):

    csv_file = "training_metrics.csv"

    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Avg Loss", "Avg PSNR (dB)", "Avg MS-SSIM"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        dataset = CompressDataset(root=args.data_dir)
    except FileNotFoundError:
        print(f"Error: Data directory '{args.data_dir}' not found!")
        return
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # print(size(loader))
    print(loader)
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

    criterion = RateDistortionLoss(lambda_rd=args.lambda_rd)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_psnr = 0.0
        total_msssim = 0.0

        progress_bar = tqdm(enumerate(loader), 
                            total=len(loader), 
                            desc=f"Epoch {epoch+1}/{args.epochs}",
                            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
        
        for batch_idx, (x, _) in progress_bar:
            x = x.to(device)
            optimizer.zero_grad()
            
            x_hat, _, logits, _ = model(x)
            loss = criterion(x_hat, x, logits)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x_denorm = (x * 0.5 + 0.5).clamp(0, 1)
                x_hat_denorm = (x_hat * 0.5 + 0.5).clamp(0, 1)
                
                batch_psnr = psnr(x_hat_denorm, x_denorm)
                batch_msssim = msssim(x_hat_denorm, x_denorm)

            total_loss += loss.item()
            total_psnr += batch_psnr
            total_msssim += batch_msssim

            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'PSNR': f"{batch_psnr:.2f}",
                'SSIM': f"{batch_msssim:.4f}"
            })

            if batch_idx % 50 == 0:
                save_reconstructed_images(x_hat[:1], epoch, batch_idx)

        avg_loss = total_loss / len(loader)
        avg_psnr = total_psnr / len(loader)
        avg_msssim = total_msssim / len(loader)

        scheduler.step()

        
        print(f"\nEpoch {epoch+1:03d} Summary | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Avg PSNR: {avg_psnr:.2f} dB | "
              f"Avg MS-SSIM: {avg_msssim:.4f}\n")
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_loss, avg_psnr, avg_msssim])

    torch.save(model.state_dict(), "./checkpoints/final_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Path to training dataset")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--lambda_rd", type=float, default=1e-4, help="Rate-distortion tradeoff")
    args = parser.parse_args()

    print("===== Training Configuration =====")
    print(f"Data directory: {args.data_dir}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Lambda RD: {args.lambda_rd}")
    print("==================================")

    try:
        train(args)
    except Exception as e:
        print(f"Error occurred during training: {str(e)}")
        raise


