from typing import Dict

import torch
import torch.nn as nn


class RetinexAdaReNet(nn.Module):
    def __init__(
        self,
        illumination: nn.Module,
        adarenet: nn.Module,
        omega: float,
        tau: float,
        eps: float,
        illum_adjust_mode: str = "gamma",
        pref_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.illumination = illumination
        self.adarenet = adarenet
        self.omega = float(omega)
        self.tau = float(tau)
        self.eps = float(eps)
        self.illum_adjust_mode = illum_adjust_mode
        self.pref_max = float(pref_max)

    def compute_illumination(self, x: torch.Tensor):
        l_t = self.illumination(x)  # 1 channel [B, 1, H, W]
        if self.illum_adjust_mode == "gamma":
            # Gamma correction: L_e = L_T^(1/omega)
            # Much better dynamic-range lifting than linear multiplication.
            # For dark pixels (L_T≈0.05, omega=15): L_e≈0.74 vs linear L_e=0.75
            l_e = torch.clamp(l_t.pow(1.0 / self.omega), 0.0, 1.0)
        else:
            # Legacy linear mode: L_e = clamp(L_T * omega, 0, 1)
            l_e = torch.clamp(l_t * self.omega, 0.0, 1.0)
        return l_t, l_e

    def compute_pref(self, x: torch.Tensor, l_t: torch.Tensor):
        l_tilde = torch.clamp(l_t, min=self.tau)
        p_ref = x / (l_tilde + self.eps)
        # Clamp P_ref to prevent numerical explosion in dark regions.
        # Without this, dark areas where L_T ≈ 0 produce P_ref >> 1,
        # amplifying noise catastrophically.
        p_ref = torch.clamp(p_ref, 0.0, self.pref_max)
        return p_ref

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        l_t, l_e = self.compute_illumination(x)
        p_ref = self.compute_pref(x, l_t)
        adain = torch.cat([p_ref, l_e], dim=1)  # [B, 4, H, W]: [P_ref(3) + L_e(1)]
        delta = self.adarenet(adain)
        r_e = p_ref - delta
        # Clamp reflectance to non-negative before reconstruction.
        # Negative R_e values cause color inversions / bizarre hues after
        # multiplication with L_e.
        r_e = torch.clamp(r_e, 0.0)
        # L_e is [B, 1, H, W], broadcast to [B, 3, H, W] for element-wise multiplication
        i_hat = torch.clamp(r_e * l_e, 0.0, 1.0)  # PyTorch broadcasts automatically
        return {
            "L_T": l_t,
            "L_e": l_e,
            "P_ref": p_ref,
            "delta": delta,
            "R_e": r_e,
            "I_hat": i_hat,
        }
