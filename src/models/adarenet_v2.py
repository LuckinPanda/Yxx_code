"""
AdaReNet V2: 改进版自适应去噪网络

改进点（保持Retinex框架和自监督算法）：
1. 残差连接 - 帮助梯度流动，防止过平滑
2. 通道注意力（SE Block）- 自适应特征加权
3. 更深的网络 - 5层而非3层
4. 特征金字塔融合 - 多尺度信息

Author: Based on AdaReNet architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(x, k, dims=(2, 3))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块"""
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        y = self.pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class ResConvBlock(nn.Module):
    """带残差连接的卷积块"""
    def __init__(self, in_ch: int, out_ch: int, use_se: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        
        # 如果通道数变化，使用1x1卷积做跳跃连接
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out = self.se(out)
        return self.act(out + identity)


class ResRotEqBlock(nn.Module):
    """带残差连接的旋转等变卷积块"""
    def __init__(self, in_ch: int, out_ch: int, use_se: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        B = x.shape[0]
        x2 = _rotate(x, 2)
        x1 = _rotate(x, 1)
        x3 = _rotate(x, 3)

        # Batch 0° + 180°
        y02 = self.conv(torch.cat([x, x2], dim=0))
        y0, y2 = y02.split(B, dim=0)

        # Batch 90° + 270°
        y13 = self.conv(torch.cat([x1, x3], dim=0))
        y1, y3 = y13.split(B, dim=0)

        # Rotate back and average
        y = (y0 + _rotate(y1, -1) + _rotate(y2, -2) + _rotate(y3, -3)) * 0.25
        y = self.se(y)
        return self.act(y + identity)


class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        # 不同尺度的感受野
        self.conv1 = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.conv3 = nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels // 4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(channels, channels // 4, kernel_size=7, padding=3)
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.conv1(x)
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)
        fused = torch.cat([f1, f3, f5, f7], dim=1)
        return self.act(self.fuse(fused) + x)


class FusionMaskNetworkV2(nn.Module):
    """改进的融合掩码网络：添加空间注意力"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.channel_mask = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, f_v: torch.Tensor, f_e: torch.Tensor) -> tuple:
        concat = torch.cat([f_v, f_e], dim=1)
        spatial_weight = self.spatial_attn(concat)  # [B, 1, H, W]
        channel_mask = self.channel_mask(concat)    # [B, C, H, W]
        return channel_mask, spatial_weight


class AdaReNetV2(nn.Module):
    """
    AdaReNet V2: 改进版自适应去噪网络
    
    改进点：
    1. 5层深度（原3层）
    2. 残差连接防止过平滑
    3. SE通道注意力
    4. 多尺度特征融合
    5. 空间注意力引导
    """
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        
        # Vanilla分支（标准卷积 + 残差）
        self.v1 = ResConvBlock(4, c, use_se=False)  # 第一层不用SE
        self.v2 = ResConvBlock(c, c)
        self.v3 = ResConvBlock(c, c)
        self.v4 = ResConvBlock(c, c)
        self.v5 = ResConvBlock(c, c)
        
        # RotEq分支（旋转等变 + 残差）
        self.e1 = ResRotEqBlock(4, c, use_se=False)
        self.e2 = ResRotEqBlock(c, c)
        self.e3 = ResRotEqBlock(c, c)
        self.e4 = ResRotEqBlock(c, c)
        self.e5 = ResRotEqBlock(c, c)
        
        # 多尺度融合
        self.ms_fusion = MultiScaleFusion(c)
        
        # 改进的融合网络
        self.fusion = FusionMaskNetworkV2(c)
        
        # 输出层
        self.out_conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Vanilla分支
        fv1 = self.v1(x)
        fv2 = self.v2(fv1)
        fv3 = self.v3(fv2)
        fv4 = self.v4(fv3)
        f_v = self.v5(fv4)
        
        # RotEq分支
        fe1 = self.e1(x)
        fe2 = self.e2(fe1)
        fe3 = self.e3(fe2)
        fe4 = self.e4(fe3)
        f_e = self.e5(fe4)
        
        # 多尺度融合
        f_v = self.ms_fusion(f_v)
        f_e = self.ms_fusion(f_e)
        
        # 自适应融合
        channel_mask, spatial_weight = self.fusion(f_v, f_e)
        f = channel_mask * f_v + (1.0 - channel_mask) * f_e
        f = f * spatial_weight  # 空间注意力加权
        
        # 输出残差
        delta = self.out_conv(f)
        return delta


class AdaReNetV2Lite(nn.Module):
    """
    AdaReNetV2轻量版：
    - 保持原始3层深度
    - 添加残差连接
    - 使用LeakyReLU替代ReLU
    - 不加SE模块（保持简单）
    """
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        
        # Vanilla分支 (添加残差连接)
        self.v1 = ResConvBlock(4, c, use_se=False)
        self.v2 = ResConvBlock(c, c, use_se=False)
        self.v3 = ResConvBlock(c, c, use_se=False)
        
        # RotEq分支 (添加残差连接)
        self.e1 = ResRotEqBlock(4, c, use_se=False)
        self.e2 = ResRotEqBlock(c, c, use_se=False)
        self.e3 = ResRotEqBlock(c, c, use_se=False)
        
        # 融合
        self.fusion_mask = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # 输出
        self.out_conv = nn.Conv2d(c, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_v = self.v3(self.v2(self.v1(x)))
        f_e = self.e3(self.e2(self.e1(x)))
        
        m = self.fusion_mask(torch.cat([f_v, f_e], dim=1))
        f = m * f_v + (1.0 - m) * f_e
        
        delta = self.out_conv(f)
        return delta


if __name__ == "__main__":
    # 测试模型
    model = AdaReNetV2(base_channels=32)
    x = torch.randn(2, 4, 256, 256)
    y = model(x)
    print(f"AdaReNetV2: input={x.shape} -> output={y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model_lite = AdaReNetV2Lite(base_channels=32)
    y_lite = model_lite(x)
    print(f"AdaReNetV2Lite: input={x.shape} -> output={y_lite.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_lite.parameters()):,}")
