"""
深度学习低光增强方法封装
Deep learning low-light enhancement method wrappers
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional


class BaseDeepEnhancer:
    """Base class for deep learning enhancement methods."""
    
    def __init__(self, name: str, device: torch.device = None):
        self.name = name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
    
    def enhance(self, img: torch.Tensor) -> torch.Tensor:
        """
        Enhance a tensor image.
        
        Args:
            img: [C, H, W] or [B, C, H, W] tensor in [0, 1]
            
        Returns:
            Enhanced tensor in same shape
        """
        raise NotImplementedError


# ============================================================
# Zero-DCE Implementation (simplified version)
# ============================================================

class ZeroDCENet(nn.Module):
    """
    Zero-DCE network architecture.
    Paper: Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
    """
    def __init__(self, n_iters: int = 8):
        super().__init__()
        self.n_iters = n_iters
        
        # DCE-Net: simple 7-layer network
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv7 = nn.Conv2d(64, 24, 3, 1, 1)  # 24 = 3 channels * 8 iterations
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # Encoder
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        
        # Decoder with skip connections
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        alpha = self.tanh(self.conv7(torch.cat([x1, x6], dim=1)))
        
        # Apply curve iteratively
        # alpha shape: [B, 24, H, W] -> split into 8 groups of 3 channels
        out = x
        for i in range(self.n_iters):
            a = alpha[:, i*3:(i+1)*3, :, :]
            out = out + a * (out - out * out)
        
        return out


class ZeroDCE(BaseDeepEnhancer):
    """Zero-DCE enhancement wrapper."""
    
    def __init__(self, ckpt_path: Optional[str] = None, device: torch.device = None):
        super().__init__("Zero-DCE", device)
        self.model = ZeroDCENet(n_iters=8).to(self.device)
        
        if ckpt_path and os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            print(f"[Zero-DCE] Loaded weights from {ckpt_path}")
        else:
            print(f"[Zero-DCE] Using randomly initialized weights (no checkpoint)")
        
        self.model.eval()
    
    @torch.no_grad()
    def enhance(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        img = img.to(self.device)
        enhanced = self.model(img)
        return enhanced.squeeze(0).clamp(0, 1)


# ============================================================
# URetinex-Net++ Wrapper
# ============================================================

class URetinexNetPP(BaseDeepEnhancer):
    """URetinex-Net++ enhancement wrapper."""
    
    def __init__(self, project_path: str = None, ratio: float = 5.0, device: torch.device = None):
        super().__init__("URetinex++", device)
        
        self.project_path = project_path or "/home/yannayanna/projects/URetinex-Net-PLUS"
        self.ratio = ratio
        self._model = None
        
    def _load_model(self):
        if self._model is not None:
            return
        
        # Add project path to sys.path temporarily
        if self.project_path not in sys.path:
            sys.path.insert(0, self.project_path)
        
        # Import required modules
        from utils import load_decom, load_AdjustFusion, load_unfolding
        from network.Math_Module import P, Q
        
        # Create options namespace
        class Opts:
            pass
        
        opts = Opts()
        opts.ratio = self.ratio
        opts.Decom_model_low_path = os.path.join(self.project_path, "pretrained_model/decom/decom_low_light.pth")
        opts.Decom_model_high_path = os.path.join(self.project_path, "pretrained_model/decom/decom_high_light.pth")
        opts.pretrain_unfolding_model_path = os.path.join(self.project_path, "pretrained_model/unfolding/unfolding_model.pth")
        opts.fusion_model_A_path = os.path.join(self.project_path, "pretrained_model/fusion_enhance/fusion.pth")
        
        # Load models
        self.unfolding_model_opts, self.model_R, self.model_L = load_unfolding(opts)
        self.fusion_opts, self.adjust_model, self.fusion_model = load_AdjustFusion(opts)
        self.model_Decom_low, self.model_Decom_high = load_decom(self.fusion_opts)
        
        self.P = P()
        self.Q = Q()
        
        self._model = True  # Mark as loaded
        print(f"[URetinex++] Loaded pretrained models from {self.project_path}")
    
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
    def enhance(self, img: torch.Tensor) -> torch.Tensor:
        self._load_model()
        
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        img = img.to(self.device)
        
        # Unfolding
        R_results, L = self._unfolding_inference(img)
        
        # Ratio
        ratio = torch.ones(L.shape, device=self.device) * self.ratio
        
        # Adjust and fusion
        High_L = self.adjust_model(l=L, alpha=ratio)
        
        if self.fusion_model is not None:
            I_enhance, _ = self.fusion_model(R_results, High_L)
        else:
            I_enhance = R_results[-1] * High_L
        
        return I_enhance.squeeze(0).clamp(0, 1)


# ============================================================
# Our Retinex-AdaReNet Wrapper
# ============================================================

class RetinexAdaReNetWrapper(BaseDeepEnhancer):
    """Retinex-AdaReNet (ours) wrapper for fair comparison."""
    
    def __init__(
        self,
        project_path: str = None,
        config_path: str = None,
        mode: str = "zero_shot",
        device: torch.device = None,
        color_correct: bool = True,
        color_strength: float = 0.3,
    ):
        super().__init__("Ours", device)
        
        self.project_path = project_path or "/home/yannayanna/projects/retinex_adarenet"
        self.config_path = config_path or os.path.join(self.project_path, "configs/infer.yaml")
        self.mode = mode
        self.color_correct = color_correct
        self.color_strength = color_strength
        self._model = None
        
    def _load_model(self):
        if self._model is not None:
            return
        
        if self.project_path not in sys.path:
            sys.path.insert(0, self.project_path)
        
        from src.models.retinex import RetinexAdaReNet
        from src.models.illumination import IlluminationNet
        from src.models.adarenet import AdaReNet
        from src.utils.config import load_config
        
        cfg = load_config(self.config_path)
        
        # Build sub-modules first (both use base_channels)
        illumination = IlluminationNet(
            base_channels=cfg["model"]["illumination_channels"],
        )
        
        adarenet = AdaReNet(
            base_channels=cfg["model"]["adarenet_channels"],
        )
        
        self._model = RetinexAdaReNet(
            illumination=illumination,
            adarenet=adarenet,
            omega=cfg["constants"]["omega"],
            tau=cfg["constants"]["tau"],
            eps=cfg["constants"]["eps"],
            illum_adjust_mode=cfg["constants"].get("illum_adjust_mode", "gamma"),
            pref_max=cfg["constants"].get("pref_max", 5.0),
        ).to(self.device)
        
        # Load checkpoints
        ckpt_cfg = cfg["ckpt"]
        self._model.illumination.load_state_dict(
            torch.load(ckpt_cfg["illum_ckpt_path"], map_location=self.device)
        )
        
        if self.mode == "zero_shot":
            denoise_ckpt = ckpt_cfg["denoise_pre_ckpt_path"]
        else:
            denoise_ckpt = ckpt_cfg["denoise_adapt_ckpt_path"]
        
        self._model.adarenet.load_state_dict(
            torch.load(denoise_ckpt, map_location=self.device)
        )
        
        self._model.eval()
        print(f"[Ours] Loaded model in {self.mode} mode")
    
    def _gray_world_correction(self, img: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """Gray-world color correction."""
        if img.dim() == 3:
            ch_mean = img.mean(dim=(1, 2), keepdim=True)
        else:
            ch_mean = img.mean(dim=(2, 3), keepdim=True)
        global_mean = ch_mean.mean(dim=-3, keepdim=True)
        gain = global_mean / (ch_mean + 1e-8)
        corrected = img * (1.0 - strength + strength * gain)
        return corrected.clamp(0.0, 1.0)
    
    @torch.no_grad()
    def enhance(self, img: torch.Tensor) -> torch.Tensor:
        self._load_model()
        
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        img = img.to(self.device)
        out = self._model(img)
        enhanced = out["I_hat"].squeeze(0)
        
        if self.color_correct and self.color_strength > 0:
            enhanced = self._gray_world_correction(enhanced, self.color_strength)
        
        return enhanced.clamp(0, 1)
