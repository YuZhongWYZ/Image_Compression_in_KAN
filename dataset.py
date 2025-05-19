import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class CompressDataset(Dataset):
    def __init__(self, root, img_size=256):
        self.root = root
        self.img_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(('png', 'jpg', 'jpeg','bmp'))]
        
        self.transform = transforms.Compose([
            transforms.Lambda(self.pad_to_multiple),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def pad_to_multiple(self, img, multiple=32):
        w, h = img.size
        new_w = ((w // multiple) + 1) * multiple if w % multiple != 0 else w
        new_h = ((h // multiple) + 1) * multiple if h % multiple != 0 else h
        padded_img = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        padded_img.paste(img, (0, 0))
        return padded_img
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        return img, img 


class ExternalImageDataset(Dataset):
    def __init__(self, data_dir, img_size=(256, 256)):

        self.data_dir = data_dir
        self.img_size = img_size
        self.image_paths = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        
        img = img.convert('RGB').resize(self.img_size[::-1])  # pillow uses (W, H) order
        
        img_array = np.array(img, dtype=np.float32)  # shape (H, W, 3)
        
        img_array = (img_array / 127.5) - 1.0
        
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (C, H, W)
        
        return tensor