import torch
import torch.nn as nn


def _rotate(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(x, k, dims=(2, 3))


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class RotEqBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rotation-equivariant response via shared weights over 0/90/180/270 rotations.
        outs = []
        for k in (0, 1, 2, 3):
            xr = _rotate(x, k)
            yr = self.conv(xr)
            y = _rotate(yr, -k)
            outs.append(y)
        y = torch.stack(outs, dim=0).mean(dim=0)
        return self.act(y)


class FusionMaskNetwork(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, f_v: torch.Tensor, f_e: torch.Tensor) -> torch.Tensor:
        # This mask is a structural gating mechanism inside AdaReNet.
        # It is NOT a blind-spot / masked-pixel training mask.
        m = self.mask(torch.cat([f_v, f_e], dim=1))
        return m


class AdaReNet(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        # Input: P_ref (3ch) + L_e (1ch) = 4 channels (per SPEC section 5.3)
        self.v1 = ConvBlock(4, base_channels)
        self.v2 = ConvBlock(base_channels, base_channels)
        self.v3 = ConvBlock(base_channels, base_channels)

        self.e1 = RotEqBlock(4, base_channels)
        self.e2 = RotEqBlock(base_channels, base_channels)
        self.e3 = RotEqBlock(base_channels, base_channels)

        self.fusion_mask = FusionMaskNetwork(base_channels)
        self.out_conv = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_v = self.v3(self.v2(self.v1(x)))
        f_e = self.e3(self.e2(self.e1(x)))
        m = self.fusion_mask(f_v, f_e)
        # M_f gates between Vanilla and EQ branches per-pixel.
        # M_f is the Fusion MaskNetwork output (structural gating) used to combine f_v and f_e.
        f = m * f_v + (1.0 - m) * f_e
        delta = self.out_conv(f)
        return delta
