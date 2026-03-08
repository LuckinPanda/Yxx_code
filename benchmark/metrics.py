"""
统一评估指标模块 - 确保所有方法使用完全相同的评估方式
Unified metrics module - ensures all methods use identical evaluation
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create 2D Gaussian window for SSIM computation."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return kernel_2d


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute PSNR (Peak Signal-to-Noise Ratio).
    
    Args:
        pred: [B, C, H, W] or [C, H, W] predicted image in [0, 1]
        target: [B, C, H, W] or [C, H, W] ground truth image in [0, 1]
        eps: epsilon for numerical stability
        
    Returns:
        PSNR value in dB (higher is better)
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    
    mse = torch.mean((pred - target) ** 2)
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr.item()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """
    Compute SSIM (Structural Similarity Index).
    
    Args:
        pred: [B, C, H, W] or [C, H, W] predicted image in [0, 1]
        target: [B, C, H, W] or [C, H, W] ground truth image in [0, 1]
        window_size: size of Gaussian window
        sigma: standard deviation of Gaussian window
        data_range: dynamic range of images
        
    Returns:
        SSIM value in [0, 1] (higher is better)
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    _, channels, _, _ = pred.shape
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    window = window.repeat(channels, 1, 1, 1)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute MAE (Mean Absolute Error).
    
    Args:
        pred: [B, C, H, W] or [C, H, W] predicted image in [0, 1]
        target: [B, C, H, W] or [C, H, W] ground truth image in [0, 1]
        
    Returns:
        MAE value in [0, 1] (lower is better)
    """
    return torch.mean(torch.abs(pred - target)).item()


class LPIPSWrapper:
    """Lazy-loaded LPIPS wrapper to avoid loading model until needed."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            import lpips
            self._model = lpips.LPIPS(net='alex').to(self.device)
            self._model.eval()
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute LPIPS perceptual loss.
        
        Args:
            pred: [B, C, H, W] or [C, H, W] predicted image in [0, 1]
            target: [B, C, H, W] or [C, H, W] ground truth image in [0, 1]
            
        Returns:
            LPIPS value (lower is better)
        """
        self._load_model()
        
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        # LPIPS expects input in [-1, 1] range
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        
        with torch.no_grad():
            lpips_value = self._model(pred_scaled, target_scaled)
        return lpips_value.item()


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_model: Optional[LPIPSWrapper] = None,
    compute_lpips: bool = True,
) -> Dict[str, float]:
    """
    Compute all metrics at once.
    
    Args:
        pred: [B, C, H, W] or [C, H, W] predicted image in [0, 1]
        target: [B, C, H, W] or [C, H, W] ground truth image in [0, 1]
        lpips_model: pre-initialized LPIPS model (optional)
        compute_lpips: whether to compute LPIPS
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'psnr': compute_psnr(pred, target),
        'ssim': compute_ssim(pred, target),
        'mae': compute_mae(pred, target),
    }
    
    if compute_lpips:
        if lpips_model is None:
            lpips_model = LPIPSWrapper(pred.device if pred.is_cuda else None)
        metrics['lpips'] = lpips_model(pred, target)
    
    return metrics
