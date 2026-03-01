"""
Comprehensive diagnostic tool for the Retinex + AdaReNet pipeline.

Prints distribution statistics for every intermediate tensor:
  L_T, L_e, P_ref, delta, R_e, I_hat
and highlights numerical-stability red flags (e.g. dark-area explosion).

Usage:
  python diagnose.py
  python diagnose.py --image /path/to/low.png
  python diagnose.py --image /path/to/low.png --high /path/to/high.png
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from src.models.illumination import IlluminationNet
from src.models.adarenet import AdaReNet
from src.models.retinex import RetinexAdaReNet
from src.utils.config import load_config


def _stat(name: str, t: torch.Tensor, percentiles=(1, 5, 50, 95, 99)):
    """Print distribution statistics for a tensor."""
    t_np = t.detach().cpu().float().numpy().ravel()
    pcts = np.percentile(t_np, percentiles)
    neg_frac = (t_np < 0).mean() * 100
    gt1_frac = (t_np > 1.0).mean() * 100
    gt5_frac = (t_np > 5.0).mean() * 100
    print(f"\n  [{name}]  shape={list(t.shape)}")
    print(f"    min={t_np.min():.6f}  mean={t_np.mean():.6f}  max={t_np.max():.6f}  std={t_np.std():.6f}")
    pct_str = "  ".join(f"p{int(p)}={v:.4f}" for p, v in zip(percentiles, pcts))
    print(f"    {pct_str}")
    if neg_frac > 0:
        print(f"    ⚠ {neg_frac:.2f}% pixels < 0")
    if gt1_frac > 0:
        print(f"    ⚠ {gt1_frac:.2f}% pixels > 1.0")
    if gt5_frac > 0:
        print(f"    ⚠ {gt5_frac:.2f}% pixels > 5.0")


def main():
    parser = argparse.ArgumentParser(description="Pipeline diagnostic tool")
    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    parser.add_argument("--image", type=str, default=None,
                        help="Single low-light image to diagnose (overrides config)")
    parser.add_argument("--high", type=str, default=None,
                        help="Optional ground-truth normal-light image for comparison")
    parser.add_argument("--ckpt_mode", type=str, choices=["pre", "adapt"], default="pre",
                        help="Which denoiser checkpoint to load")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    c = cfg["constants"]
    tau = c["tau"]
    eps = c["eps"]
    omega = c["omega"]
    pref_max = c.get("pref_max", 5.0)
    illum_adjust_mode = c.get("illum_adjust_mode", "gamma")

    illum = IlluminationNet(base_channels=cfg["model"]["illumination_channels"])
    adarenet = AdaReNet(base_channels=cfg["model"]["adarenet_channels"])
    model = RetinexAdaReNet(illum, adarenet, omega=omega, tau=tau, eps=eps,
                            illum_adjust_mode=illum_adjust_mode, pref_max=pref_max)
    model.to(device)

    # Load checkpoints
    illum_ckpt = cfg["ckpt"]["illum_ckpt_path"]
    if args.ckpt_mode == "adapt":
        denoise_ckpt = cfg["ckpt"]["denoise_adapt_ckpt_path"]
    else:
        denoise_ckpt = cfg["ckpt"]["denoise_pre_ckpt_path"]

    model.illumination.load_state_dict(torch.load(illum_ckpt, map_location=device))
    model.adarenet.load_state_dict(torch.load(denoise_ckpt, map_location=device))
    model.eval()

    # Load image
    if args.image:
        img_path = args.image
    else:
        # Use first image in test_low_dir
        test_dir = Path(cfg["data"]["test_low_dir"])
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        imgs = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in exts)
        if not imgs:
            print(f"No images found in {test_dir}")
            return
        img_path = str(imgs[0])

    low = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    low_tensor = torch.from_numpy(low).permute(2, 0, 1).unsqueeze(0).to(device)

    print("=" * 70)
    print("  RETINEX + AdaReNet  PIPELINE DIAGNOSTIC")
    print("=" * 70)
    print(f"\n  Config: tau={tau}, eps={eps}, omega={omega}, pref_max={pref_max}")
    print(f"  Illum adjust: {illum_adjust_mode}")
    print(f"  Image: {img_path}")
    print(f"  Denoise ckpt: {denoise_ckpt}")

    with torch.no_grad():
        # Step-by-step forward with diagnostics
        _stat("Input (I)", low_tensor)

        l_t, l_e = model.compute_illumination(low_tensor)
        _stat("L_T (raw illumination)", l_t)
        _stat("L_e (enhanced illumination)", l_e)

        # Check dark area fraction
        lt_np = l_t.cpu().numpy().ravel()
        dark_frac = (lt_np < tau).mean() * 100
        very_dark_frac = (lt_np < 0.01).mean() * 100
        print(f"\n  🔍 L_T < tau ({tau}): {dark_frac:.1f}% of pixels (will be clamped)")
        print(f"  🔍 L_T < 0.01: {very_dark_frac:.1f}% of pixels")

        # P_ref WITHOUT clamping (diagnostic only)
        l_tilde_raw = torch.clamp(l_t, min=tau)
        p_ref_raw = low_tensor / (l_tilde_raw + eps)
        _stat("P_ref (before pref_max clamp)", p_ref_raw)

        # P_ref WITH clamping (actual pipeline)
        p_ref = model.compute_pref(low_tensor, l_t)
        _stat("P_ref (after pref_max clamp)", p_ref)
        clipped_frac = (p_ref_raw.cpu() > pref_max).float().mean() * 100
        print(f"  🔍 Pixels clipped by pref_max: {clipped_frac:.1f}%")

        # AdaReNet
        adain = torch.cat([p_ref, l_e], dim=1)
        delta = model.adarenet(adain)
        _stat("delta (AdaReNet residual)", delta)

        r_e = torch.clamp(p_ref - delta, 0.0)
        _stat("R_e (cleaned reflectance)", r_e)

        i_hat = torch.clamp(r_e * l_e, 0.0, 1.0)
        _stat("I_hat (final output)", i_hat)

    # Channel-wise analysis for color bias detection
    print("\n" + "-" * 70)
    print("  COLOR BALANCE ANALYSIS")
    print("-" * 70)
    for name, t in [("Input", low_tensor), ("P_ref", p_ref), ("R_e", r_e),
                     ("delta", delta), ("I_hat", i_hat)]:
        t_cpu = t[0].cpu()
        r, g, b = t_cpu[0].mean().item(), t_cpu[1].mean().item(), t_cpu[2].mean().item()
        print(f"  {name:12s}  R={r:.4f}  G={g:.4f}  B={b:.4f}  "
              f"(G-R={g-r:+.4f}, G-B={g-b:+.4f})")

    # Ground truth comparison
    if args.high:
        high = np.array(Image.open(args.high).convert("RGB")).astype(np.float32) / 255.0
        high_tensor = torch.from_numpy(high).permute(2, 0, 1).unsqueeze(0).to(device)
        _stat("Ground Truth (high)", high_tensor)
        mse = ((i_hat - high_tensor) ** 2).mean().item()
        psnr = 10 * np.log10(1.0 / (mse + 1e-8))
        print(f"\n  📊 PSNR = {psnr:.2f} dB")
        print(f"  📊 Mean diff (I_hat - GT) = {(i_hat - high_tensor).mean().item():.4f}")

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
