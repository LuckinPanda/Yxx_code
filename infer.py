import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LowLightDataset
from src.models.adarenet import AdaReNet
from src.models.illumination import IlluminationNet
from src.models.retinex import RetinexAdaReNet
from src.utils.config import load_config
from src.utils.image import save_image


def make_comparison(low: torch.Tensor, enhanced: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Create a side-by-side comparison image: low | enhanced | high.

    Args:
        low: [3, H, W] low-light input
        enhanced: [3, H, W] model output
        high: [3, H, W] ground truth normal-light

    Returns:
        [3, H, W*3 + 4] concatenated image with 2px white separator
    """
    _, H, W = low.shape
    sep = torch.ones(3, H, 2, device=low.device)  # white separator
    return torch.cat([low, sep, enhanced, sep, high], dim=2)


def _eq(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) < tol


def validate_constants(cfg: dict) -> None:
    # Note: omega can vary for better illumination enhancement
    # tau is now configurable (recommended 0.05-0.2 for numerical stability)
    c = cfg["constants"]
    if not _eq(c["eps"], 1e-6):
        raise ValueError("SPEC-fixed constant eps must be 1e-6.")
    tau = c["tau"]
    if tau < 0.01:
        print(f"[WARN] tau={tau} is very small. Dark-area P_ref may explode. Recommend tau>=0.05.")
    elif tau > 0.5:
        print(f"[WARN] tau={tau} is very large. Over-clamping may lose dark-area detail.")


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return kernel_2d


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
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
    return ssim_map.mean()


def compute_lpips(pred: torch.Tensor, target: torch.Tensor, lpips_model) -> float:
    """Compute LPIPS perceptual loss."""
    # LPIPS expects input in [-1, 1] range
    pred_scaled = pred * 2.0 - 1.0
    target_scaled = target * 2.0 - 1.0
    with torch.no_grad():
        lpips_value = lpips_model(pred_scaled, target_scaled)
    return lpips_value.item()


def save_metrics_csv(metrics_list: list, output_path: Path) -> None:
    """Save per-image metrics to CSV file."""
    if not metrics_list:
        return
    
    fieldnames = metrics_list[0].keys()
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)


def compute_statistics(metrics_list: list) -> dict:
    """Compute statistics from per-image metrics."""
    if not metrics_list:
        return {}
    
    psnrs = [m['psnr'] for m in metrics_list]
    ssims = [m['ssim'] for m in metrics_list]
    lpips_values = [m['lpips'] for m in metrics_list if 'lpips' in m]
    
    # Sort for worst-case analysis
    psnrs_sorted = sorted(psnrs)
    ssims_sorted = sorted(ssims)
    lpips_sorted = sorted(lpips_values, reverse=True) if lpips_values else []
    
    # Compute worst 10%
    n_worst = max(1, int(len(psnrs) * 0.1))
    
    stats = {
        'count': len(metrics_list),
        'psnr': {
            'mean': np.mean(psnrs),
            'std': np.std(psnrs),
            'median': np.median(psnrs),
            'min': np.min(psnrs),
            'max': np.max(psnrs),
            'worst_10pct_mean': np.mean(psnrs_sorted[:n_worst]),
        },
        'ssim': {
            'mean': np.mean(ssims),
            'std': np.std(ssims),
            'median': np.median(ssims),
            'min': np.min(ssims),
            'max': np.max(ssims),
            'worst_10pct_mean': np.mean(ssims_sorted[:n_worst]),
        },
    }
    
    if lpips_values:
        stats['lpips'] = {
            'mean': np.mean(lpips_values),
            'std': np.std(lpips_values),
            'median': np.median(lpips_values),
            'min': np.min(lpips_values),
            'max': np.max(lpips_values),
            'worst_10pct_mean': np.mean(lpips_sorted[:n_worst]),  # worst = highest LPIPS
        }
    
    return stats


def save_summary_json(stats: dict, output_path: Path, metadata: dict = None) -> None:
    """Save summary statistics to JSON file."""
    output_data = {
        'metadata': metadata or {},
        'statistics': stats
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    parser.add_argument("--mode", type=str, choices=["zero_shot", "adapt"], default="zero_shot")
    parser.add_argument("--disable_adarenet", action="store_true", help="Bypass AdaReNet (delta=0) for debugging")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--compute_lpips", action="store_true", help="Compute LPIPS metric (requires lpips package)")
    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        print(f"[Infer] Random seed set to: {args.seed}")

    cfg = load_config(args.config)
    validate_constants(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize LPIPS model if requested
    lpips_model = None
    if args.compute_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='alex').to(device)
            lpips_model.eval()
            print("[Infer] LPIPS model loaded (AlexNet backbone)")
        except ImportError:
            print("[Infer] WARNING: lpips package not found. Install with: pip install lpips")
            print("[Infer] Skipping LPIPS computation")
            args.compute_lpips = False

    data_cfg = cfg["data"]
    resize = data_cfg["resize"]
    if resize is not None:
        resize = tuple(resize)

    test_low_dir = data_cfg["test_low_dir"]
    test_high_dir = data_cfg.get("test_high_dir")
    if test_high_dir:
        dataset = LowLightDataset(
            mode="paired_by_index",
            source_low_dir=test_low_dir,
            source_high_dir=test_high_dir,
            target_low_dir=None,
            resize=resize,
        )
    else:
        dataset = LowLightDataset(
            mode="target_low_only",
            source_low_dir=None,
            source_high_dir=None,
            target_low_dir=test_low_dir,
            resize=resize,
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    illum = IlluminationNet(base_channels=cfg["model"]["illumination_channels"])
    adarenet = AdaReNet(base_channels=cfg["model"]["adarenet_channels"])

    illum_adjust_mode = cfg["constants"].get("illum_adjust_mode", "gamma")
    pref_max = cfg["constants"].get("pref_max", 5.0)
    model = RetinexAdaReNet(
        illum,
        adarenet,
        omega=cfg["constants"]["omega"],
        tau=cfg["constants"]["tau"],
        eps=cfg["constants"]["eps"],
        illum_adjust_mode=illum_adjust_mode,
        pref_max=pref_max,
    ).to(device)

    ckpt_cfg = cfg["ckpt"]
    illum_ckpt = ckpt_cfg["illum_ckpt_path"]
    model.illumination.load_state_dict(torch.load(illum_ckpt, map_location=device))

    if args.mode == "zero_shot":
        denoise_ckpt = ckpt_cfg["denoise_pre_ckpt_path"]
    else:
        denoise_ckpt = ckpt_cfg["denoise_adapt_ckpt_path"]
    print(f"[Infer] mode={args.mode} denoise_ckpt={denoise_ckpt}")
    model.adarenet.load_state_dict(torch.load(denoise_ckpt, map_location=device))
    if args.disable_adarenet:
        print("[Infer] AdaReNet disabled (delta=0)")

    model.eval()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = Path(data_cfg["output_dir"])
    if args.seed is not None:
        out_dir = base_out_dir / f"{args.mode}_{timestamp}_seed{args.seed}"
    else:
        out_dir = base_out_dir / f"{args.mode}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = out_dir / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Infer] Output directory: {out_dir}")

    # Metrics tracking
    metrics_list = []

    # Metrics tracking
    metrics_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Infer ({args.mode})", ncols=80):
            low = batch["low"].to(device, non_blocking=True)
            name = batch["name"][0]
            
            if args.disable_adarenet:
                l_t, l_e = model.compute_illumination(low)
                p_ref = model.compute_pref(low, l_t)
                i_hat = torch.clamp(p_ref * l_e, 0.0, 1.0)[0]
            else:
                out = model(low)
                i_hat = out["I_hat"][0]
            
            save_image(i_hat, str(out_dir / name))

            # Compute metrics if ground truth available
            if "high" in batch:
                high = batch["high"].to(device, non_blocking=True)

                # Save comparison image: low | enhanced | ground truth
                comparison = make_comparison(
                    low[0].clamp(0, 1), i_hat.clamp(0, 1), high[0].clamp(0, 1)
                )
                save_image(comparison, str(compare_dir / name))

                pred = i_hat.unsqueeze(0)
                target = high
                
                psnr = compute_psnr(pred, target).item()
                ssim = compute_ssim(pred, target).item()
                
                metric_dict = {
                    'image_name': name,
                    'psnr': psnr,
                    'ssim': ssim,
                }
                
                # Compute LPIPS if requested
                if args.compute_lpips and lpips_model is not None:
                    lpips_val = compute_lpips(pred, target, lpips_model)
                    metric_dict['lpips'] = lpips_val
                
                metrics_list.append(metric_dict)

    print(f"[Infer] Saved {len(loader)} images to: {out_dir}")
    
    # Save and display metrics
    if metrics_list:
        # Save per-image metrics to CSV
        csv_path = out_dir / "metrics_per_image.csv"
        save_metrics_csv(metrics_list, csv_path)
        print(f"[Infer] Saved per-image metrics to: {csv_path}")
        
        # Compute statistics
        stats = compute_statistics(metrics_list)
        
        # Save summary statistics to JSON
        metadata = {
            'mode': args.mode,
            'timestamp': timestamp,
            'seed': args.seed,
            'disable_adarenet': args.disable_adarenet,
            'compute_lpips': args.compute_lpips,
            'denoise_ckpt': denoise_ckpt,
        }
        json_path = out_dir / "metrics_summary.json"
        save_summary_json(stats, json_path, metadata)
        print(f"[Infer] Saved summary statistics to: {json_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print(f"EVALUATION SUMMARY ({args.mode})")
        print("="*60)
        print(f"Total images: {stats['count']}")
        print(f"\nPSNR (dB):")
        print(f"  Mean:    {stats['psnr']['mean']:.4f} ± {stats['psnr']['std']:.4f}")
        print(f"  Median:  {stats['psnr']['median']:.4f}")
        print(f"  Range:   [{stats['psnr']['min']:.4f}, {stats['psnr']['max']:.4f}]")
        print(f"  Worst 10%: {stats['psnr']['worst_10pct_mean']:.4f}")
        
        print(f"\nSSIM:")
        print(f"  Mean:    {stats['ssim']['mean']:.4f} ± {stats['ssim']['std']:.4f}")
        print(f"  Median:  {stats['ssim']['median']:.4f}")
        print(f"  Range:   [{stats['ssim']['min']:.4f}, {stats['ssim']['max']:.4f}]")
        print(f"  Worst 10%: {stats['ssim']['worst_10pct_mean']:.4f}")
        
        if 'lpips' in stats:
            print(f"\nLPIPS (lower is better):")
            print(f"  Mean:    {stats['lpips']['mean']:.4f} ± {stats['lpips']['std']:.4f}")
            print(f"  Median:  {stats['lpips']['median']:.4f}")
            print(f"  Range:   [{stats['lpips']['min']:.4f}, {stats['lpips']['max']:.4f}]")
            print(f"  Worst 10%: {stats['lpips']['worst_10pct_mean']:.4f}")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
