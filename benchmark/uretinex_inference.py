#!/usr/bin/env python
"""
URetinex-Net++ 独立推理脚本
用于基准测试，与项目解耦
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# 添加 URetinex-Net++ 项目路径
URETINEX_PATH = "/home/yannayanna/projects/URetinex-Net-PLUS"
sys.path.insert(0, URETINEX_PATH)

from network.Math_Module import P, Q


def load_decom_fixed(decom_low_path, decom_high_path=None):
    """Fixed version of load_decom that uses explicit paths."""
    from network.decom import Decom
    
    def create_and_load(model_path):
        model = Decom()
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        # The checkpoint has nested structure: state_dict['model_R']
        if 'state_dict' in ckpt:
            # Check if it's the nested format
            if 'model_R' in ckpt['state_dict']:
                model.load_state_dict(ckpt['state_dict']['model_R'])
            else:
                model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    decom_low_model = create_and_load(decom_low_path)
    decom_high_model = None
    if decom_high_path and os.path.exists(decom_high_path):
        decom_high_model = create_and_load(decom_high_path)
    
    return decom_low_model, decom_high_model


def load_unfolding_fixed(unfolding_path):
    """Fixed version of load_unfolding."""
    from utils import define_modelR, define_modelL
    
    checkpoint = torch.load(unfolding_path, map_location='cpu', weights_only=False)
    old_opts = checkpoint["opts"]
    model_R = define_modelR(old_opts)
    model_L = define_modelL(old_opts)
    model_R.load_state_dict(checkpoint['state_dict']['model_R'])
    model_L.load_state_dict(checkpoint['state_dict']['model_L'])
    
    for param in model_R.parameters():
        param.requires_grad = False
    for param in model_L.parameters():
        param.requires_grad = False
    
    return old_opts, model_R, model_L


def load_adjust_fusion_fixed(fusion_path):
    """Fixed version of load_AdjustFusion."""
    from utils import define_modelA, define_compositor
    
    checkpoint = torch.load(fusion_path, map_location='cpu', weights_only=False)
    opts = checkpoint["opts"]
    model_A = define_modelA(opts)
    model_A.load_state_dict(checkpoint['state_dict']['model_A'])
    
    for param in model_A.parameters():
        param.requires_grad = False
    
    model_fusion = None
    if "fusion_model" in opts.__dict__ if hasattr(opts, '__dict__') else opts:
        model_fusion = define_compositor(opts)
        if model_fusion is not None:
            model_fusion.load_state_dict(checkpoint['state_dict']['model_compositor'])
            for param in model_fusion.parameters():
                param.requires_grad = False
    
    return opts, model_A, model_fusion


class URetinexPPInference:
    def __init__(self, ratio: float = 5.0, device: str = 'cuda'):
        self.ratio = ratio
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Define paths
        decom_low_path = os.path.join(URETINEX_PATH, "pretrained_model/decom/decom_low_light.pth")
        decom_high_path = os.path.join(URETINEX_PATH, "pretrained_model/decom/decom_high_light.pth")
        unfolding_path = os.path.join(URETINEX_PATH, "pretrained_model/unfolding/unfolding_model.pth")
        fusion_path = os.path.join(URETINEX_PATH, "pretrained_model/fusion_enhance/fusion.pth")
        
        # Load models with fixed loaders
        print("[URetinex++] Loading models...")
        self.unfolding_model_opts, self.model_R, self.model_L = load_unfolding_fixed(unfolding_path)
        self.fusion_opts, self.adjust_model, self.fusion_model = load_adjust_fusion_fixed(fusion_path)
        self.model_Decom_low, self.model_Decom_high = load_decom_fixed(decom_low_path, decom_high_path)
        
        self.P = P()
        self.Q = Q()
        
        # Move to device
        self.model_R = self.model_R.to(self.device)
        self.model_L = self.model_L.to(self.device)
        self.adjust_model = self.adjust_model.to(self.device)
        self.model_Decom_low = self.model_Decom_low.to(self.device)
        if self.fusion_model is not None:
            self.fusion_model = self.fusion_model.to(self.device)
        
        self.transform = transforms.ToTensor()
        print("[URetinex++] Models loaded successfully")
    
    def _unfolding_inference(self, input_low_img):
        """Run unfolding network."""
        P_results, Q_results, R_results, L_results = [[], [], [], []]
        
        for t in range(self.unfolding_model_opts.round):
            if t == 0:
                P, Q = self.model_Decom_low(input_low_img)
            else:
                w_p = (self.unfolding_model_opts.gamma + self.unfolding_model_opts.Roffset * t)
                w_q = (self.unfolding_model_opts.lamda + self.unfolding_model_opts.Loffset * t)
                P = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q = self.Q(I=input_low_img, P=P, L=L, lamda=w_q)
            
            R = self.model_R(r=P, l=Q)
            L = self.model_L(l=Q)
            R_results.append(R)
            L_results.append(L)
        
        results = []
        for t in range(len(R_results)):
            if (t+1) in self.fusion_opts.fusion_layers:
                results.append(R_results[t])
        
        return results, L_results[-1]
    
    @torch.no_grad()
    def enhance(self, img_tensor):
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
        
        # Unfolding
        R_results, L = self._unfolding_inference(img_tensor)
        
        # Ratio
        ratio = torch.ones(L.shape, device=self.device) * self.ratio
        
        # Adjust and fusion
        High_L = self.adjust_model(l=L, alpha=ratio)
        
        if self.fusion_model is not None:
            I_enhance, _ = self.fusion_model(R_results, High_L)
        else:
            I_enhance = R_results[-1] * High_L
        
        return I_enhance.squeeze(0).clamp(0, 1)
    
    def process_directory(self, input_dir, output_dir):
        """Process all images in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        image_files = [f for f in image_files if f.is_file()]
        
        print(f"[URetinex++] Processing {len(image_files)} images...")
        
        for img_path in tqdm(image_files, desc="URetinex++"):
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            
            # Enhance
            enhanced = self.enhance(img_tensor)
            
            # Save
            enhanced_np = enhanced.cpu().permute(1, 2, 0).numpy()
            enhanced_np = (np.clip(enhanced_np, 0, 1) * 255).astype(np.uint8)
            out_path = output_dir / img_path.name
            Image.fromarray(enhanced_np).save(str(out_path))
        
        print(f"[URetinex++] Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ratio', type=float, default=5.0)
    args = parser.parse_args()
    
    model = URetinexPPInference(ratio=args.ratio)
    model.process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
