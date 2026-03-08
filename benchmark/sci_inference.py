#!/usr/bin/env python
"""
SCI (Self-Calibrated Illumination) 推理脚本
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


class EnhanceNetwork(nn.Module):
    """SCI 的增强网络（从原项目复制以避免 loss.py 依赖）"""
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)

        return illu


class SCIModel(nn.Module):
    """SCI 推理模型"""
    def __init__(self):
        super(SCIModel, self).__init__()
        self.enhance = EnhanceNetwork(layers=1, channels=3)

    def forward(self, input):
        i = self.enhance(input)
        r = input / i
        r = torch.clamp(r, 0, 1)
        return i, r


class SCIInference:
    def __init__(self, ckpt_path: str = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if ckpt_path is None:
            # 默认使用 medium.pt
            ckpt_path = "/home/yannayanna/projects/SCI/CVPR/weights/medium.pt"
        
        print(f"[SCI] Loading model from {ckpt_path}")
        self.model = SCIModel().to(self.device)
        
        # 加载预训练权重
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        
        # 只加载 enhance 网络的权重
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        
        self.model.eval()
        print("[SCI] Model loaded successfully")
    
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
        
        # SCI 返回 (illumination, reflectance)
        _, enhanced = self.model(img_tensor)
        
        return enhanced.squeeze(0).clamp(0, 1)
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all images in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        image_files = [f for f in image_files if f.is_file()]
        
        print(f"[SCI] Processing {len(image_files)} images...")
        
        for img_path in tqdm(image_files, desc="SCI"):
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
        
        print(f"[SCI] Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None, help="weights path (easy.pt, medium.pt, difficult.pt)")
    args = parser.parse_args()
    
    model = SCIInference(ckpt_path=args.ckpt)
    model.process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
