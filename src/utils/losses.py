"""
Detail-Preserving Loss Functions for Self-Supervised Denoising

针对 over-smoothing 问题的多种损失函数设计：
1. 多尺度梯度损失 - 保护不同尺度的边缘
2. 拉普拉斯金字塔损失 - 保护高频纹理
3. 局部对比度损失 - 保持局部细节差异
4. 频域损失 - 显式保护高频分量
5. 结构相似性损失 - 结构级别的保护
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 多尺度梯度损失 (Multi-Scale Gradient Loss)
# ═══════════════════════════════════════════════════════════════════════════════

def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Basic gradient-domain L1 loss for edge preservation."""
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


def multiscale_gradient_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    scales: List[int] = [1, 2, 4]
) -> torch.Tensor:
    """
    多尺度梯度损失：在多个下采样尺度上计算梯度损失
    
    Args:
        pred: [B, C, H, W] 预测图像
        target: [B, C, H, W] 目标图像
        scales: 下采样因子列表
    
    Returns:
        加权平均的多尺度梯度损失
    """
    total_loss = 0.0
    weights = [1.0, 0.5, 0.25]  # 更高分辨率权重更大
    
    for i, scale in enumerate(scales):
        if scale == 1:
            p, t = pred, target
        else:
            p = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
            t = F.avg_pool2d(target, kernel_size=scale, stride=scale)
        
        weight = weights[i] if i < len(weights) else weights[-1]
        total_loss += weight * gradient_loss(p, t)
    
    return total_loss / sum(weights[:len(scales)])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 拉普拉斯金字塔损失 (Laplacian Pyramid Loss)
# ═══════════════════════════════════════════════════════════════════════════════

class LaplacianPyramid(nn.Module):
    """构建拉普拉斯金字塔，分离不同频率成分"""
    
    def __init__(self, levels: int = 3):
        super().__init__()
        self.levels = levels
        # Gaussian kernel for downsampling
        kernel = torch.tensor([1., 4., 6., 4., 1.]) / 16.0
        kernel = kernel.view(1, 1, 5, 1) * kernel.view(1, 1, 1, 5)
        self.register_buffer('kernel', kernel)
    
    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian 下采样"""
        B, C, H, W = x.shape
        kernel = self.kernel.expand(C, 1, 5, 5)
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        x = F.conv2d(x, kernel, groups=C, stride=2)
        return x
    
    def _upsample(self, x: torch.Tensor, size: tuple) -> torch.Tensor:
        """双线性上采样"""
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        构建拉普拉斯金字塔
        
        Returns:
            金字塔层列表 [high_freq_1, high_freq_2, ..., low_freq_residual]
        """
        pyramid = []
        current = x
        
        for _ in range(self.levels - 1):
            down = self._downsample(current)
            up = self._upsample(down, current.shape[2:])
            # 拉普拉斯层 = 当前 - 上采样的下一层（即高频残差）
            laplacian = current - up
            pyramid.append(laplacian)
            current = down
        
        # 最后一层是低频残差
        pyramid.append(current)
        return pyramid


def laplacian_pyramid_loss(
    pred: torch.Tensor, 
    target: torch.Tensor,
    levels: int = 3,
    weights: Optional[List[float]] = None
) -> torch.Tensor:
    """
    拉普拉斯金字塔损失：对高频层给予更高权重
    
    这个损失函数特别适合解决 over-smoothing，因为：
    - 显式分离并保护高频成分
    - 高频层（纹理、细节）被单独监督
    """
    if weights is None:
        # 高频层（索引越小频率越高）权重递减
        # 第一层（最高频）权重最大
        weights = [2.0, 1.0, 0.5][:levels]
    
    lap = LaplacianPyramid(levels)
    pred_pyr = lap(pred)
    tgt_pyr = lap(target)
    
    total_loss = 0.0
    for i, (p, t) in enumerate(zip(pred_pyr, tgt_pyr)):
        w = weights[i] if i < len(weights) else weights[-1]
        total_loss += w * F.l1_loss(p, t)
    
    return total_loss / sum(weights)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 局部对比度损失 (Local Contrast Loss)
# ═══════════════════════════════════════════════════════════════════════════════

def local_contrast_loss(
    pred: torch.Tensor, 
    target: torch.Tensor,
    kernel_size: int = 7
) -> torch.Tensor:
    """
    局部对比度损失：保持局部标准差（纹理丰富区域有更高的局部标准差）
    
    over-smoothing 会降低局部对比度，这个损失可以防止这种情况。
    """
    padding = kernel_size // 2
    
    # 计算局部均值
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=pred.device) / (kernel_size ** 2)
    
    def local_std(x: torch.Tensor) -> torch.Tensor:
        """计算局部标准差"""
        B, C, H, W = x.shape
        # 逐通道计算
        std_list = []
        for c in range(C):
            xc = x[:, c:c+1, :, :]
            local_mean = F.conv2d(xc, kernel, padding=padding)
            local_sq_mean = F.conv2d(xc ** 2, kernel, padding=padding)
            local_var = local_sq_mean - local_mean ** 2
            local_var = torch.clamp(local_var, min=1e-6)
            std_list.append(torch.sqrt(local_var))
        return torch.cat(std_list, dim=1)
    
    pred_std = local_std(pred)
    tgt_std = local_std(target)
    
    return F.l1_loss(pred_std, tgt_std)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 高频损失 (High-Frequency Loss) - 使用 Sobel/Laplacian
# ═══════════════════════════════════════════════════════════════════════════════

class HighFrequencyExtractor(nn.Module):
    """提取高频成分（边缘、纹理）"""
    
    def __init__(self, mode: str = 'laplacian'):
        super().__init__()
        self.mode = mode
        
        if mode == 'laplacian':
            # Laplacian kernel
            kernel = torch.tensor([
                [0., -1., 0.],
                [-1., 4., -1.],
                [0., -1., 0.]
            ]).view(1, 1, 3, 3)
        elif mode == 'sobel':
            # Sobel kernels for both directions
            sobel_x = torch.tensor([
                [-1., 0., 1.],
                [-2., 0., 2.],
                [-1., 0., 1.]
            ]).view(1, 1, 3, 3)
            sobel_y = sobel_x.transpose(2, 3)
            kernel = torch.cat([sobel_x, sobel_y], dim=0)  # [2, 1, 3, 3]
        elif mode == 'log':
            # Laplacian of Gaussian (LoG) - 更好地检测细节
            kernel = torch.tensor([
                [0., 0., -1., 0., 0.],
                [0., -1., -2., -1., 0.],
                [-1., -2., 16., -2., -1.],
                [0., -1., -2., -1., 0.],
                [0., 0., -1., 0., 0.]
            ]).view(1, 1, 5, 5)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.register_buffer('kernel', kernel)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取高频成分"""
        B, C, H, W = x.shape
        
        if self.mode == 'sobel':
            # Sobel: 返回梯度幅度
            hf_list = []
            for c in range(C):
                xc = x[:, c:c+1, :, :]
                grad = F.conv2d(xc, self.kernel, padding=1)  # [B, 2, H, W]
                magnitude = torch.sqrt(grad[:, 0:1] ** 2 + grad[:, 1:2] ** 2 + 1e-6)
                hf_list.append(magnitude)
            return torch.cat(hf_list, dim=1)
        else:
            # Laplacian / LoG
            pad = self.kernel.shape[-1] // 2
            hf_list = []
            for c in range(C):
                xc = x[:, c:c+1, :, :]
                hf = F.conv2d(xc, self.kernel, padding=pad)
                hf_list.append(hf)
            return torch.cat(hf_list, dim=1)


def high_frequency_loss(
    pred: torch.Tensor, 
    target: torch.Tensor,
    mode: str = 'laplacian'
) -> torch.Tensor:
    """
    高频损失：比较高频成分
    
    Args:
        mode: 'laplacian', 'sobel', 或 'log'
    """
    extractor = HighFrequencyExtractor(mode).to(pred.device)
    pred_hf = extractor(pred)
    tgt_hf = extractor(target)
    return F.l1_loss(pred_hf, tgt_hf)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SSIM 损失 (Structural Similarity Loss)
# ═══════════════════════════════════════════════════════════════════════════════

def ssim_loss(
    pred: torch.Tensor, 
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2
) -> torch.Tensor:
    """
    SSIM 损失：1 - SSIM
    
    SSIM 考虑亮度、对比度、结构三方面，对局部结构变化敏感
    """
    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = g.view(1, -1) * g.view(-1, 1)
    window = window / window.sum()
    window = window.view(1, 1, window_size, window_size)
    
    C = pred.shape[1]
    window = window.expand(C, 1, window_size, window_size)
    
    pad = window_size // 2
    
    mu1 = F.conv2d(pred, window, padding=pad, groups=C)
    mu2 = F.conv2d(target, window, padding=pad, groups=C)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred ** 2, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu12
    
    # SSIM 公式
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return 1.0 - ssim_map.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 综合细节保持损失 (Combined Detail-Preserving Loss)
# ═══════════════════════════════════════════════════════════════════════════════

class DetailPreservingLoss(nn.Module):
    """
    综合细节保持损失
    
    结合多种损失函数，全面保护图像细节：
    - 拉普拉斯金字塔损失：分层保护不同频率
    - 高频损失：显式保护边缘和纹理
    - 局部对比度损失：保持纹理丰富度
    """
    
    def __init__(
        self,
        use_laplacian_pyramid: bool = True,
        use_high_freq: bool = True,
        use_local_contrast: bool = False,  # 计算开销较大，默认关闭
        use_ssim: bool = False,
        laplacian_levels: int = 3,
        hf_mode: str = 'laplacian',
        weights: dict = None
    ):
        super().__init__()
        self.use_laplacian_pyramid = use_laplacian_pyramid
        self.use_high_freq = use_high_freq
        self.use_local_contrast = use_local_contrast
        self.use_ssim = use_ssim
        self.laplacian_levels = laplacian_levels
        self.hf_mode = hf_mode
        
        # 默认权重
        self.weights = weights or {
            'laplacian_pyramid': 0.3,
            'high_freq': 0.2,
            'local_contrast': 0.1,
            'ssim': 0.2
        }
        
        if use_laplacian_pyramid:
            self.lap_pyramid = LaplacianPyramid(laplacian_levels)
        if use_high_freq:
            self.hf_extractor = HighFrequencyExtractor(hf_mode)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        计算综合细节保持损失
        
        Returns:
            dict: 包含总损失和各分量损失
        """
        losses = {}
        total = 0.0
        
        if self.use_laplacian_pyramid:
            loss_lap = laplacian_pyramid_loss(pred, target, self.laplacian_levels)
            losses['laplacian_pyramid'] = loss_lap
            total += self.weights['laplacian_pyramid'] * loss_lap
        
        if self.use_high_freq:
            # Move extractor to same device
            if self.hf_extractor.kernel.device != pred.device:
                self.hf_extractor = self.hf_extractor.to(pred.device)
            pred_hf = self.hf_extractor(pred)
            tgt_hf = self.hf_extractor(target)
            loss_hf = F.l1_loss(pred_hf, tgt_hf)
            losses['high_freq'] = loss_hf
            total += self.weights['high_freq'] * loss_hf
        
        if self.use_local_contrast:
            loss_lc = local_contrast_loss(pred, target)
            losses['local_contrast'] = loss_lc
            total += self.weights['local_contrast'] * loss_lc
        
        if self.use_ssim:
            loss_ssim = ssim_loss(pred, target)
            losses['ssim'] = loss_ssim
            total += self.weights['ssim'] * loss_ssim
        
        losses['total'] = total
        return losses


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Edge-Aware 自适应权重（用于 Mask Loss）
# ═══════════════════════════════════════════════════════════════════════════════

def compute_edge_weight(
    target: torch.Tensor, 
    min_weight: float = 1.0,
    max_weight: float = 5.0
) -> torch.Tensor:
    """
    计算边缘感知的像素级权重
    
    在边缘/纹理区域给予更高权重，平坦区域给予较低权重
    这样可以让网络更关注细节区域的重建
    """
    # 计算梯度幅度
    dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    # Pad 回原始尺寸
    dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
    dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
    
    # 梯度幅度
    grad_mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
    grad_mag = grad_mag.mean(dim=1, keepdim=True)  # 跨通道平均
    
    # 归一化到 [min_weight, max_weight]
    grad_max = grad_mag.amax(dim=(2, 3), keepdim=True) + 1e-6
    grad_norm = grad_mag / grad_max
    
    weight = min_weight + (max_weight - min_weight) * grad_norm
    return weight


# ═══════════════════════════════════════════════════════════════════════════════
# Color consistency loss (保持原有)
# ═══════════════════════════════════════════════════════════════════════════════

def color_consistency_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Channel-ratio preservation loss to prevent color shift."""
    pred_sum = pred.sum(dim=1, keepdim=True) + eps
    tgt_sum = target.sum(dim=1, keepdim=True) + eps
    return F.l1_loss(pred / pred_sum, target / tgt_sum)
