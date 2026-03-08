"""
改进的损失函数（保持自监督框架）

新增损失：
1. SSIM损失 - 结构保持
2. 多尺度梯度损失 - 多级边缘保持
3. 平滑度正则 - 更好的去噪效果
4. 感知一致性 - 高频细节保持

Author: Based on original self-supervised training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window_1d(size: int, sigma: float) -> torch.Tensor:
    """生成1D高斯窗口"""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_window_2d(size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    """生成2D高斯窗口用于SSIM计算"""
    g = _gaussian_window_1d(size, sigma)
    g2d = g[:, None] * g[None, :]
    g2d = g2d.view(1, 1, size, size).repeat(channels, 1, 1, 1)
    return g2d.to(device)


class SSIMLoss(nn.Module):
    """SSIM损失 - 结构相似性损失"""
    def __init__(self, window_size: int = 11, sigma: float = 1.5, reduction: str = 'mean'):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.reduction = reduction
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self._window = None
        self._channels = None

    def _get_window(self, channels: int, device: torch.device) -> torch.Tensor:
        if self._window is None or self._channels != channels:
            self._window = _gaussian_window_2d(self.window_size, self.sigma, channels, device)
            self._channels = channels
        return self._window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.shape[1]
        window = self._get_window(channels, pred.device)
        pad = self.window_size // 2

        mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
        mu2 = F.conv2d(target, window, padding=pad, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        
        ssim_val = ssim_map.mean(dim=(1, 2, 3))
        
        if self.reduction == 'mean':
            return 1 - ssim_val.mean()
        elif self.reduction == 'sum':
            return (1 - ssim_val).sum()
        else:
            return 1 - ssim_val


class MultiScaleGradientLoss(nn.Module):
    """多尺度梯度损失 - 在多个分辨率上保持边缘结构"""
    def __init__(self, scales: int = 3, weights: list = None):
        super().__init__()
        self.scales = scales
        self.weights = weights or [1.0 / (2 ** i) for i in range(scales)]  # 金字塔权重

    def _gradient(self, x: torch.Tensor) -> tuple:
        """计算水平和垂直梯度"""
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        pred_scale = pred
        target_scale = target
        
        for i in range(self.scales):
            pred_dx, pred_dy = self._gradient(pred_scale)
            target_dx, target_dy = self._gradient(target_scale)
            
            loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
            total_loss += self.weights[i] * loss
            
            # 下采样到下一个尺度
            if i < self.scales - 1:
                pred_scale = F.avg_pool2d(pred_scale, 2)
                target_scale = F.avg_pool2d(target_scale, 2)
        
        return total_loss


class SmoothnessLoss(nn.Module):
    """平滑度正则 - 鼓励piece-wise smooth的输出"""
    def __init__(self, edge_aware: bool = True):
        super().__init__()
        self.edge_aware = edge_aware

    def forward(self, pred: torch.Tensor, guidance: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: 预测的reflectance
            guidance: 引导图像（可选，用于边缘感知平滑）
        """
        dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        if self.edge_aware and guidance is not None:
            # 边缘感知：在引导图像边缘处允许更大的梯度
            guide_dx = (guidance[:, :, :, 1:] - guidance[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
            guide_dy = (guidance[:, :, 1:, :] - guidance[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
            
            # 指数衰减权重
            weight_x = torch.exp(-5.0 * guide_dx)
            weight_y = torch.exp(-5.0 * guide_dy)
            
            loss = (weight_x * dx.abs()).mean() + (weight_y * dy.abs()).mean()
        else:
            loss = dx.abs().mean() + dy.abs().mean()
        
        return loss


class FrequencyLoss(nn.Module):
    """频域损失 - 保持高频细节"""
    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT变换
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        
        # 幅度谱
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # 高频区域掩码（简单的距离中心越远权重越大）
        B, C, H, W = pred.shape
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=pred.device).float() - cy
        x = torch.arange(W, device=pred.device).float() - cx
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(xx ** 2 + yy ** 2) / max(H, W)
        high_freq_weight = torch.clamp(dist, 0, 1).view(1, 1, H, W)
        
        # 高频损失（加权的频域差异）
        loss = (high_freq_weight * (pred_mag - target_mag).abs()).mean()
        return loss


class LocalContrastLoss(nn.Module):
    """局部对比度损失 - 保持局部结构"""
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 计算局部均值
        pred_mean = F.avg_pool2d(pred, self.kernel_size, stride=1, padding=self.padding)
        target_mean = F.avg_pool2d(target, self.kernel_size, stride=1, padding=self.padding)
        
        # 局部对比度 = 局部标准差的近似
        pred_contrast = (pred - pred_mean).abs()
        target_contrast = (target - target_mean).abs()
        
        # 对比度一致性损失
        return F.l1_loss(pred_contrast, target_contrast)


class CombinedLoss(nn.Module):
    """
    组合损失 - 用于自监督去噪训练
    
    保持原有自监督框架，添加结构保持损失
    """
    def __init__(
        self,
        use_ssim: bool = True,
        use_multiscale_grad: bool = True,
        use_smoothness: bool = False,
        use_frequency: bool = False,
        use_local_contrast: bool = False,
        lambda_ssim: float = 0.5,
        lambda_grad: float = 0.1,
        lambda_smooth: float = 0.01,
        lambda_freq: float = 0.1,
        lambda_contrast: float = 0.1,
    ):
        super().__init__()
        self.use_ssim = use_ssim
        self.use_multiscale_grad = use_multiscale_grad
        self.use_smoothness = use_smoothness
        self.use_frequency = use_frequency
        self.use_local_contrast = use_local_contrast
        
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad
        self.lambda_smooth = lambda_smooth
        self.lambda_freq = lambda_freq
        self.lambda_contrast = lambda_contrast
        
        if use_ssim:
            self.ssim_loss = SSIMLoss()
        if use_multiscale_grad:
            self.grad_loss = MultiScaleGradientLoss(scales=3)
        if use_smoothness:
            self.smooth_loss = SmoothnessLoss(edge_aware=True)
        if use_frequency:
            self.freq_loss = FrequencyLoss()
        if use_local_contrast:
            self.contrast_loss = LocalContrastLoss()

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor = None,
        guidance: torch.Tensor = None
    ) -> dict:
        """
        Args:
            pred: 预测的reflectance
            target: 目标reflectance (P_ref)
            mask: 自监督掩码M
            guidance: 引导图像（用于边缘感知平滑）
            
        Returns:
            包含各项损失的字典
        """
        losses = {}
        
        # 基础L1/masked L1损失仍在训练脚本中计算
        
        if self.use_ssim:
            losses['ssim'] = self.lambda_ssim * self.ssim_loss(pred, target)
            
        if self.use_multiscale_grad:
            losses['grad'] = self.lambda_grad * self.grad_loss(pred, target)
            
        if self.use_smoothness:
            losses['smooth'] = self.lambda_smooth * self.smooth_loss(pred, guidance)
            
        if self.use_frequency:
            losses['freq'] = self.lambda_freq * self.freq_loss(pred, target)
            
        if self.use_local_contrast:
            losses['contrast'] = self.lambda_contrast * self.contrast_loss(pred, target)
        
        return losses


def color_consistency_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """通道比例保持损失 - 防止色偏"""
    pred_sum = pred.sum(dim=1, keepdim=True) + eps
    tgt_sum = target.sum(dim=1, keepdim=True) + eps
    return F.l1_loss(pred / pred_sum, target / tgt_sum)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """基础梯度域L1损失"""
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


if __name__ == "__main__":
    # 测试损失函数
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    
    ssim_loss = SSIMLoss()
    print(f"SSIM Loss: {ssim_loss(pred, target).item():.6f}")
    
    grad_loss = MultiScaleGradientLoss()
    print(f"MultiScale Gradient Loss: {grad_loss(pred, target).item():.6f}")
    
    smooth_loss = SmoothnessLoss()
    print(f"Smoothness Loss: {smooth_loss(pred).item():.6f}")
    
    combined = CombinedLoss(use_ssim=True, use_multiscale_grad=True)
    losses = combined(pred, target)
    print(f"Combined losses: {losses}")
