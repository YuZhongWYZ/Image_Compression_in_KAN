import torch
import torch.nn as nn
from kan import KAN,KANConv2DLayer

class KANEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Stage 1: 3x256x256 -> 64x128x128
            KANConv2DLayer(3, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            
            # Stage 2: 64x128x128 -> 128x64x64
            KANConv2DLayer(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            
            KANConv2DLayer(128, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.feature_compressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*32, 1024),  
            nn.ReLU()
        )
        
        self.kan = KAN(
            layers_hidden=[1024, 512],  
            grid_size=3
        )
        
        self.shape_adjust = nn.Sequential(
            nn.Linear(512, 256*8*8),
            nn.Unflatten(1, (256, 8, 8))
        )

    def forward(self, x):
        x = self.conv_layers(x)         # [B,32,32,32]
        x = self.feature_compressor(x)  # [B,1024]
        x = self.kan(x)                 # [B,512]
        return self.shape_adjust(x)     # [B,256,8,8]



class QuantizationAgent(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.logit_generator = nn.Sequential(
            KANConv2DLayer(channels, 128, 3, padding=1),
            nn.ReLU(),
            KANConv2DLayer(128, channels, 3, padding=1)
        )
        
        self.rate_estimator = nn.Sequential(
            KANConv2DLayer(channels, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, y):

        logits = self.logit_generator(y)  # 生成量化参数
        value = torch.round(y + logits) - logits.detach()  # 直通梯度
        rate = self.rate_estimator(y)     # 码率估计
        return logits, value, rate

class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

class KANCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = KANEncoder()
        self.quant_agent = QuantizationAgent()
        self.decoder = self._build_decoder()

    def _build_decoder(self):
        return nn.Sequential(
            KANConv2DLayer(256, 128, 3, padding=1),  # [B,128,8,8]
            
            Lambda(lambda x: x.permute(0, 2, 3, 1).reshape(-1, 128)),  # [B*8*8, 128]
            KAN(layers_hidden=[128, 128], grid_size=3),
            Lambda(lambda x: x.reshape(-1, 8, 8, 128).permute(0, 3, 1, 2)),  # [B,128,8,8]
            
            nn.Upsample(scale_factor=2, mode='nearest'),  # [B,128,16,16]
            KANConv2DLayer(128, 64, 3, padding=1),  # [B,64,16,16]
            
            Lambda(lambda x: x.permute(0, 2, 3, 1).reshape(-1, 64)),
            KAN(layers_hidden=[64, 64], grid_size=3), 
            Lambda(lambda x: x.reshape(-1, 16, 16, 64).permute(0, 3, 1, 2)),
            
            nn.Upsample(scale_factor=2, mode='nearest'),  # [B,64,32,32]
            KANConv2DLayer(64, 32, 3, padding=1),  # [B,32,32,32]
            
            Lambda(lambda x: x.permute(0, 2, 3, 1).reshape(-1, 32)),
            KAN(layers_hidden=[32, 32], grid_size=3),
            Lambda(lambda x: x.reshape(-1, 32, 32, 32).permute(0, 3, 1, 2)),
            
            nn.Upsample(scale_factor=2, mode='nearest'),  # [B,32,64,64]
            KANConv2DLayer(32, 16, 3, padding=1),  # [B,16,64,64]
            
            Lambda(lambda x: x.permute(0, 2, 3, 1).reshape(-1, 16)),
            KAN(layers_hidden=[16, 16], grid_size=3),
            Lambda(lambda x: x.reshape(-1, 64, 64, 16).permute(0, 3, 1, 2)),
            
            KANConv2DLayer(16, 3, 5, padding=2),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Tanh()
        )
    
    def forward(self, x):
        y = self.encoder(x)  # output shape: 256x8x8
        
        logits, y_quant, rate = self.quant_agent(y)
        
        x_hat = self.decoder(y_quant)
        return x_hat, y_quant, logits, rate

    def _quantize(self, y, logits):
        scales = torch.softmax(logits, dim=1)[:, 1].view(-1, 1, 1, 1)
        return y + (torch.rand_like(y) - 0.5) * scales
