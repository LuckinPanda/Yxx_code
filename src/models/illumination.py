import torch
import torch.nn as nn


class IlluminationNet(nn.Module):
    """Illumination estimation network with dilated convolutions.

    Uses increasing dilation rates + residual connections to capture
    large-scale illumination structure while preserving local detail.
    Effective receptive field ~67 pixels (vs ~7 in the 3-layer version).

    Output: L_T ∈ [0, 1], shape [B, 1, H, W].
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        bc = base_channels
        # Input projection
        self.input_conv = nn.Conv2d(3, bc, kernel_size=3, padding=1)
        # Feature extraction with increasing dilation for large receptive field
        self.conv1 = nn.Conv2d(bc, bc, kernel_size=3, padding=1)                # dilation=1
        self.conv2 = nn.Conv2d(bc, bc, kernel_size=3, padding=2, dilation=2)    # dilation=2
        self.conv3 = nn.Conv2d(bc, bc, kernel_size=3, padding=4, dilation=4)    # dilation=4
        self.conv4 = nn.Conv2d(bc, bc, kernel_size=3, padding=8, dilation=8)    # dilation=8
        self.conv5 = nn.Conv2d(bc, bc, kernel_size=3, padding=1)                # dilation=1
        # Output projection (1×1 conv for channel reduction)
        self.out_conv = nn.Conv2d(bc, 1, kernel_size=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.act(self.input_conv(x))
        f1 = self.act(self.conv1(f0))
        f2 = self.act(self.conv2(f1)) + f1      # residual
        f3 = self.act(self.conv3(f2)) + f2      # residual
        f4 = self.act(self.conv4(f3)) + f3      # residual
        f5 = self.act(self.conv5(f4)) + f0      # skip to input features
        return self.sigmoid(self.out_conv(f5))
