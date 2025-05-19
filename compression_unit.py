from dahuffman import HuffmanCodec
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class CompressionUnit(nn.Module):
    def __init__(self, feat_dim=1024, groups=64, codebook_size=256):
        super().__init__()
        assert feat_dim % groups == 0
        self.feat_dim = feat_dim
        self.groups = groups
        self.codebook_size = codebook_size
        self.group_dim = feat_dim // groups
        
        self.codebooks = nn.Parameter(torch.randn(groups, codebook_size, self.group_dim))
        self.context_model = nn.Sequential(
            nn.Conv2d(groups, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, groups * codebook_size, 1)
        )
        self.sparsity_controller = nn.Sequential(
            nn.Conv2d(feat_dim, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )

    def quantize(self, x, target_sparsity=0.5):
        B, C, H, W = x.shape
        assert C == self.groups * self.group_dim
        
        x_grouped = x.view(B, self.groups, self.group_dim, H, W)
        x_permuted = x_grouped.permute(0, 1, 3, 4, 2).contiguous()
        x_flat = x_permuted.view(B, self.groups, H*W, self.group_dim)
        codebooks_exp = self.codebooks.unsqueeze(0)
        distances = torch.cdist(x_flat, codebooks_exp.squeeze(0), p=2)
        distances = distances.view(B, self.groups, H, W, -1)
        raw_indices = torch.argmin(distances, dim=-1)
        
        importance = self.sparsity_controller(x)  # [B,1,H,W]
        importance_exp = importance.view(B, 1, H, W).expand(-1, self.groups, -1, -1)
        
        keep_num = max(1, int(H * W * (1 - target_sparsity)))
        mask = torch.zeros_like(raw_indices, dtype=torch.bool)
        
        for b in range(B):
            for g in range(self.groups):
                group_imp = importance_exp[b, g].view(-1).cpu()
                _, topk_idx = torch.topk(group_imp, k=keep_num)
                topk_idx = topk_idx % (H*W)  
                mask[b, g].view(-1)[topk_idx] = True
        
        masked_indices = raw_indices * mask.long()
        codebooks_exp_batch = self.codebooks.unsqueeze(0).unsqueeze(3).unsqueeze(4).expand(B, -1, -1, H, W, -1)
        indices_exp = masked_indices.unsqueeze(2).unsqueeze(-1).expand(-1, -1, -1, -1,-1, self.group_dim)
        quantized = torch.gather(codebooks_exp_batch, dim=2, index=indices_exp)
        quantized = quantized.squeeze(2).permute(0, 4, 1, 2, 3).contiguous().view(B, -1, H, W)
        quantized = x + (quantized - x).detach()
        
        return raw_indices, quantized, mask

    def get_probs(self, indices):
        B, G, H, W = indices.shape
        logits = self.context_model(indices.float())
        logits = logits.view(B, G, self.codebook_size, H, W)
        return torch.softmax(logits, dim=2).permute(0, 1, 3, 4, 2)

    def forward(self, x, target_sparsity=0.5):
        raw_indices, quantized, mask = self.quantize(x, target_sparsity)
        probs = self.get_probs(raw_indices)
        prob_loss = F.cross_entropy(
            probs.reshape(-1, self.codebook_size),
            raw_indices.reshape(-1).long()
        )
        commit_loss = F.mse_loss(quantized, x.detach())
        return quantized, commit_loss + 0.1 * prob_loss

    def compress(self, x, target_sparsity=0.5):
        raw_indices, quantized, mask = self.quantize(x, target_sparsity)
        masked_indices = raw_indices * mask.long() 
    
        indices_np = masked_indices.cpu().numpy().astype(np.int16)
        mask_np = mask.cpu().numpy().astype(bool)  

        indices_flat = indices_np.ravel()
        mask_flat = mask_np.ravel()

        nonzero_values = indices_flat[mask_flat].tolist() 

        expected_num = mask_flat.sum()
        actual_num = len(nonzero_values)
        assert actual_num == expected_num, f"nonzero values count mismatch: {actual_num} vs {expected_num}"

        codec = HuffmanCodec.from_data(nonzero_values)
        encoded_values = codec.encode(nonzero_values)

        from itertools import groupby
        mask_rle = [(k, sum(1 for _ in g)) for k, g in groupby(mask_flat)]

        meta = {
            'codec': codec,
            'values': encoded_values,
            'mask_rle': mask_rle,
            'shape': indices_np.shape,
            'dtype': str(indices_np.dtype)
        }
        return pickle.dumps(meta)


    def decompress(self, byte_stream):
        meta = pickle.loads(byte_stream)
        codec = meta['codec']
        values = meta['values']
        mask_rle = meta['mask_rle']
        shape = meta['shape']
        
        def rle_decode(rle_data, target_size):
            decoded = []
            for k, count in rle_data:
                decoded.extend([k] * count)
            return np.array(decoded[:target_size], dtype=np.uint8)
        
        total_elements = np.prod(shape)
        mask_flat = rle_decode(mask_rle, total_elements)
        
        indices_flat = np.zeros(total_elements, dtype=np.int64)
        nonzero_values = codec.decode(values)
        assert len(nonzero_values) == mask_flat.sum(),
        indices_flat[mask_flat.astype(bool)] = nonzero_values
        
        indices = indices_flat.reshape(shape)
        indices = torch.from_numpy(indices).to(self.codebooks.device)
        
        B, G, H, W = shape
        codebooks_exp = self.codebooks.unsqueeze(0).unsqueeze(3).unsqueeze(4).expand(B, -1, -1, H, W, -1)
        indices_exp = indices.unsqueeze(2).unsqueeze(-1).expand(-1, -1, -1, -1,-1, self.group_dim)
        quantized = torch.gather(codebooks_exp, dim=2, index=indices_exp)
        quantized = quantized.squeeze(2).permute(0, 4, 1, 2, 3).contiguous().view(B, -1, H, W)
        return quantized

def plot_features(original, recon, title):
    plt.figure(figsize=(10, 4))
    
    original_np = original.detach().cpu().numpy()
    recon_np = recon.detach().cpu().numpy()
    
    channel = np.random.randint(0, original_np.shape[1])
    
    original_feature = original_np[0, channel]
    recon_feature = recon_np[0, channel]
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_feature, cmap='viridis')
    plt.title("Original Feature")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(recon_feature, cmap='viridis')
    plt.title(title)
    plt.axis('off')
    
    plt.show()


if __name__ == "__main__":
    comp_unit = CompressionUnit(feat_dim=1024)
    x = torch.randn(16, 1024, 8, 8)  # test input
    
    print("=== test ===")
    sparsity_levels = [0.2, 0.5, 0.8]
    
    for sparsity in sparsity_levels:
        byte_stream = comp_unit.compress(x, target_sparsity=sparsity)
        recon = comp_unit.decompress(byte_stream)
        
        # calculate metrics
        original_size = x.element_size() * x.numel()
        compressed_size = len(byte_stream)
        mse = F.mse_loss(recon, x).item()
        
        print(f"target sparsity: {sparsity:.1f}")
        print(f"  - compressed rate: {original_size/compressed_size:.1f}x")
        print(f"  - rebuild MSE: {mse:.4f}\n")
        
        plot_features(x, recon, f"Sparsity={sparsity}")
