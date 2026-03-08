"""
传统低光增强方法 - 作为基线对比
Traditional low-light enhancement methods as baselines
"""

import cv2
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Optional


class BaseEnhancer(ABC):
    """Base class for all enhancement methods."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def enhance(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance a low-light image.
        
        Args:
            img: [H, W, 3] image in [0, 1] float32, RGB order
            
        Returns:
            Enhanced image [H, W, 3] in [0, 1] float32, RGB order
        """
        pass
    
    def enhance_tensor(self, img: torch.Tensor) -> torch.Tensor:
        """
        Enhance a tensor image.
        
        Args:
            img: [C, H, W] or [B, C, H, W] tensor in [0, 1]
            
        Returns:
            Enhanced tensor in same shape
        """
        if img.dim() == 4:
            results = []
            for i in range(img.shape[0]):
                results.append(self.enhance_tensor(img[i]))
            return torch.stack(results, dim=0)
        
        # [C, H, W] -> [H, W, C] numpy
        img_np = img.permute(1, 2, 0).cpu().numpy()
        enhanced_np = self.enhance(img_np)
        # [H, W, C] -> [C, H, W] tensor
        enhanced = torch.from_numpy(enhanced_np).permute(2, 0, 1)
        return enhanced.to(img.device, img.dtype)


class HistogramEqualization(BaseEnhancer):
    """Histogram Equalization enhancement."""
    
    def __init__(self):
        super().__init__("HE")
    
    def enhance(self, img: np.ndarray) -> np.ndarray:
        # Convert to uint8 for OpenCV
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        # Convert RGB to YUV
        yuv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YUV)
        
        # Apply histogram equalization to Y channel
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        
        # Convert back to RGB
        enhanced_u8 = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        
        return enhanced_u8.astype(np.float32) / 255.0


class CLAHE(BaseEnhancer):
    """Contrast Limited Adaptive Histogram Equalization."""
    
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        super().__init__("CLAHE")
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def enhance(self, img: np.ndarray) -> np.ndarray:
        # Convert to uint8 for OpenCV
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        # Convert RGB to LAB
        lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
        
        # Create CLAHE object and apply to L channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced_u8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_u8.astype(np.float32) / 255.0


class GammaCorrection(BaseEnhancer):
    """Adaptive Gamma Correction for dark images."""
    
    def __init__(self, gamma: Optional[float] = None, adaptive: bool = True):
        super().__init__("Gamma")
        self.gamma = gamma
        self.adaptive = adaptive
    
    def _compute_adaptive_gamma(self, img: np.ndarray) -> float:
        """Compute adaptive gamma based on image brightness."""
        mean_brightness = np.mean(img)
        # Map brightness to gamma: darker images get lower gamma (more brightening)
        # gamma = 0.4 - 0.8 for typical low-light images
        gamma = 0.4 + 0.4 * mean_brightness
        return max(0.2, min(1.0, gamma))
    
    def enhance(self, img: np.ndarray) -> np.ndarray:
        if self.adaptive and self.gamma is None:
            gamma = self._compute_adaptive_gamma(img)
        else:
            gamma = self.gamma or 0.5
        
        # Apply gamma correction
        enhanced = np.power(np.clip(img, 0, 1), gamma)
        return enhanced.astype(np.float32)


class RetinexMSR(BaseEnhancer):
    """Multi-Scale Retinex enhancement."""
    
    def __init__(self, sigma_list: list = [15, 80, 250], low_clip: float = 0.01, high_clip: float = 0.99):
        super().__init__("MSR")
        self.sigma_list = sigma_list
        self.low_clip = low_clip
        self.high_clip = high_clip
    
    def _single_scale_retinex(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """Single scale retinex."""
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        return np.log10(img + 1e-8) - np.log10(blur + 1e-8)
    
    def enhance(self, img: np.ndarray) -> np.ndarray:
        # Multi-scale retinex
        msr = np.zeros_like(img)
        for sigma in self.sigma_list:
            msr += self._single_scale_retinex(img, sigma)
        msr = msr / len(self.sigma_list)
        
        # Normalize to [0, 1]
        for c in range(3):
            msr[:, :, c] = self._normalize_channel(msr[:, :, c])
        
        return msr.astype(np.float32)
    
    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Normalize channel to [0, 1] with percentile clipping."""
        low = np.percentile(channel, self.low_clip * 100)
        high = np.percentile(channel, self.high_clip * 100)
        channel = (channel - low) / (high - low + 1e-8)
        return np.clip(channel, 0, 1)


class Identity(BaseEnhancer):
    """No enhancement - just return input (for testing)."""
    
    def __init__(self):
        super().__init__("Input")
    
    def enhance(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32)


def get_traditional_methods() -> dict:
    """Get all traditional enhancement methods."""
    return {
        'HE': HistogramEqualization(),
        'CLAHE': CLAHE(),
        'Gamma': GammaCorrection(),
        'MSR': RetinexMSR(),
    }
