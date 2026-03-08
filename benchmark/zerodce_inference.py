#!/usr/bin/env python
"""
Zero-DCE 推理脚本
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 添加 Zero-DCE 项目路径
ZERODCE_PATH = "/home/yannayanna/projects/Zero-DCE/Zero-DCE_code"
sys.path.insert(0, ZERODCE_PATH)

from model import enhance_net_nopool


class ZeroDCEInference:
    def __init__(self, ckpt_path: str = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if ckpt_path is None:
            ckpt_path = os.path.join(ZERODCE_PATH, "snapshots/Epoch99.pth")
        
        print(f"[Zero-DCE] Loading model from {ckpt_path}")
        self.model = enhance_net_nopool().to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
        self.model.eval()
        print("[Zero-DCE] Model loaded successfully")
    
    @torch.no_grad()
    def enhance(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Enhance a single image tensor.
        
        Args:
            img_tensor: [C, H, W] or [B, C, H, W] in [0, 1]
            
        Returns:
            Enhanced tensor [C, H, W] in [0, 1]
        """
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        img_tensor = img_tensor.to(self.device)
        
        # Zero-DCE outputs: (enhance_image_1, enhance_image, r)
        _, enhanced, _ = self.model(img_tensor)
        
        return enhanced.squeeze(0).clamp(0, 1)
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all images in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        image_files = [f for f in image_files if f.is_file()]
        
        print(f"[Zero-DCE] Processing {len(image_files)} images...")
        
        for img_path in tqdm(image_files, desc="Zero-DCE"):
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            
            # Enhance
            enhanced = self.enhance(img_tensor)
            
            # Save
            enhanced_np = enhanced.cpu().permute(1, 2, 0).numpy()
            enhanced_np = (np.clip(enhanced_np, 0, 1) * 255).astype(np.uint8)
            out_path = output_dir / img_path.name
            Image.fromarray(enhanced_np).save(str(out_path))
        
        print(f"[Zero-DCE] Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()
    
    model = ZeroDCEInference(ckpt_path=args.ckpt)
    model.process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
