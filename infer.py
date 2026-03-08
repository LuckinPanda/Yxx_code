import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter as scipy_gaussian
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LowLightDataset
from src.models.adarenet import AdaReNet
from src.models.adarenet_v2 import AdaReNetV2Lite
from src.models.illumination import IlluminationNet
from src.models.retinex import RetinexAdaReNet
from src.utils.config import load_config
from src.utils.image import save_image


def gray_world_correction(img: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Gray-world color correction to reduce systematic color cast.

    Assumes the average color of a scene should be achromatic (gray).
    Scales each channel so that channel means become equal.

    Args:
        img: [3, H, W] or [B, 3, H, W] image in [0, 1]
        strength: blending factor (0 = no correction, 1 = full correction)

    Returns:
        Color-corrected image in [0, 1]
    """
    if img.dim() == 3:
        ch_mean = img.mean(dim=(1, 2), keepdim=True)  # [3, 1, 1]
    else:
        ch_mean = img.mean(dim=(2, 3), keepdim=True)  # [B, 3, 1, 1]
    global_mean = ch_mean.mean(dim=-3, keepdim=True)
    gain = global_mean / (ch_mean + 1e-8)
    corrected = img * (1.0 - strength + strength * gain)
    return corrected.clamp(0.0, 1.0)


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


# ─────────────────────────────────────────────────────────────
# Post-processing functions (inference-time only, training unaffected)
# ─────────────────────────────────────────────────────────────

def bilateral_denoise_np(img_np: np.ndarray, d: int = 9,
                         sigma_color: float = 30, sigma_space: float = 9) -> np.ndarray:
    """Bilateral filter: smooths noise/color artifacts while preserving edges.
    Input/output: [H,W,3] float32 in [0,1]."""
    img_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(img_u8, d, sigma_color, sigma_space)
    return filtered.astype(np.float32) / 255.0


def guided_filter_np(img_np: np.ndarray, radius: int = 8, eps: float = 0.01,
                     guide: np.ndarray = None) -> np.ndarray:
    """Guided filter: edge-preserving smoothing with better structure preservation.
    
    Unlike bilateral filter, guided filter can better preserve fine structures
    and gradients while removing noise. Particularly effective for low-light
    enhancement where we want to preserve texture details.
    
    Args:
        img_np: [H,W,3] float32 in [0,1] - image to filter
        radius: filter window radius (default 8, larger = smoother)
        eps: regularization (default 0.01, smaller = sharper edges)
        guide: optional guide image, if None uses img as guide
    
    Returns:
        [H,W,3] float32 filtered image
    """
    from cv2 import ximgproc
    
    img_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    if guide is not None:
        guide_u8 = (np.clip(guide, 0, 1) * 255).astype(np.uint8)
    else:
        guide_u8 = img_u8
    
    # Process each channel separately with grayscale guide
    result = np.zeros_like(img_u8)
    guide_gray = cv2.cvtColor(guide_u8, cv2.COLOR_RGB2GRAY)
    for c in range(3):
        result[:, :, c] = ximgproc.guidedFilter(
            guide_gray, img_u8[:, :, c], radius, eps * 255**2
        )
    
    return result.astype(np.float32) / 255.0


def nlmeans_denoise_np(img_np: np.ndarray, h: float = 10,
                       template_size: int = 7, search_size: int = 21) -> np.ndarray:
    """Non-local means denoising: globally searches for similar patches.
    
    More aggressive denoising than bilateral filter, better at removing
    color noise while preserving textures.
    
    Args:
        img_np: [H,W,3] float32 in [0,1]
        h: filter strength (higher = more smoothing, 10-15 for moderate noise)
        template_size: patch size for comparison
        search_size: search window size
    
    Returns:
        [H,W,3] float32 denoised image
    """
    img_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(
        img_u8, None, h, h, template_size, search_size
    )
    return denoised.astype(np.float32) / 255.0


def auto_contrast_np(img_np: np.ndarray, clip_pct: float = 1.0) -> np.ndarray:
    """Percentile-based contrast stretching.
    Clips extreme pixel values then rescales to [0,1].
    Input/output: [H,W,3] float32."""
    low_val = np.percentile(img_np, clip_pct)
    high_val = np.percentile(img_np, 100 - clip_pct)
    if high_val - low_val < 1e-6:
        return img_np
    stretched = (img_np - low_val) / (high_val - low_val)
    return np.clip(stretched, 0, 1).astype(np.float32)


def unsharp_mask_np(img_np: np.ndarray, sigma: float = 1.0,
                    strength: float = 0.3) -> np.ndarray:
    """Unsharp masking for edge enhancement.
    Input/output: [H,W,3] float32 in [0,1]."""
    blurred = scipy_gaussian(img_np, sigma=[sigma, sigma, 0])
    sharpened = img_np + strength * (img_np - blurred)
    return np.clip(sharpened, 0, 1).astype(np.float32)


def _make_gaussian_kernel_1d(sigma: float, device, dtype):
    """Create 1D Gaussian kernel for separable convolution."""
    ks = int(6 * sigma + 1) | 1  # ensure odd, ≥ 6σ+1
    if ks < 3:
        ks = 3
    half = ks // 2
    coords = torch.arange(ks, device=device, dtype=dtype) - half
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def smooth_illumination_forward(model, x: torch.Tensor,
                                smooth_sigma: float = 3.0) -> torch.Tensor:
    """Enhanced forward pass: Gaussian-smooth L_T before P_ref computation.
    Smoother illumination → less noise amplification in P_ref → cleaner output.
    Also eliminates per-pixel brightness fluctuations.
    Returns: [3,H,W] enhanced image (single image, no batch dim)."""
    l_t, _ = model.compute_illumination(x)

    # Separable Gaussian blur on L_T [B,1,H,W]
    g = _make_gaussian_kernel_1d(smooth_sigma, x.device, x.dtype)
    pad_h = len(g) // 2
    l_t_s = F.conv2d(l_t, g.view(1, 1, -1, 1), padding=(pad_h, 0))
    l_t_s = F.conv2d(l_t_s, g.view(1, 1, 1, -1), padding=(0, pad_h))

    # Recompute L_e from smoothed L_T
    if model.illum_adjust_mode == "gamma":
        l_e = torch.clamp(l_t_s.pow(1.0 / model.omega), 0.0, 1.0)
    else:
        l_e = torch.clamp(l_t_s * model.omega, 0.0, 1.0)

    # P_ref with smoothed L_T (less noise amplification)
    l_tilde = torch.clamp(l_t_s, min=model.tau)
    p_ref = torch.clamp(x / (l_tilde + model.eps), 0.0, model.pref_max)

    # Run denoiser
    delta = model.adarenet(torch.cat([p_ref, l_e], dim=1))
    r_e = torch.clamp(p_ref - delta, 0.0)
    i_hat = torch.clamp(r_e * l_e, 0.0, 1.0)
    return i_hat[0]  # [3,H,W]


def postprocess_tensor(img: torch.Tensor, bilateral: bool = True,
                       contrast: bool = True, sharpen: bool = True,
                       bilateral_d: int = 9, bilateral_sc: float = 30,
                       bilateral_ss: float = 9, contrast_pct: float = 1.0,
                       sharpen_sigma: float = 1.0,
                       sharpen_strength: float = 0.3,
                       denoise_method: str = "bilateral",
                       guided_radius: int = 8, guided_eps: float = 0.01,
                       nlm_h: float = 10) -> torch.Tensor:
    """Apply post-processing pipeline to [3,H,W] tensor in [0,1].
    
    Args:
        img: Input image tensor
        bilateral: Enable denoising (controlled by denoise_method)
        denoise_method: "bilateral" | "guided" | "nlmeans" | "combo"
            - bilateral: Fast edge-preserving smoothing
            - guided: Better structure preservation
            - nlmeans: Best at removing color noise
            - combo: guided + bilateral for maximum quality
        guided_radius: Radius for guided filter (default 8)
        guided_eps: Regularization for guided filter (default 0.01)
        nlm_h: Strength for non-local means (default 10)
    """
    device = img.device
    img_np = img.cpu().numpy().transpose(1, 2, 0)  # [H,W,3]
    
    if bilateral:
        if denoise_method == "bilateral":
            img_np = bilateral_denoise_np(img_np, d=bilateral_d,
                                           sigma_color=bilateral_sc,
                                           sigma_space=bilateral_ss)
        elif denoise_method == "guided":
            img_np = guided_filter_np(img_np, radius=guided_radius, eps=guided_eps)
        elif denoise_method == "nlmeans":
            img_np = nlmeans_denoise_np(img_np, h=nlm_h)
        elif denoise_method == "combo":
            # Combined: guided filter first for structure, then mild bilateral
            img_np = guided_filter_np(img_np, radius=guided_radius, eps=guided_eps)
            img_np = bilateral_denoise_np(img_np, d=5, sigma_color=50, sigma_space=5)
    
    if contrast:
        img_np = auto_contrast_np(img_np, clip_pct=contrast_pct)
    if sharpen:
        img_np = unsharp_mask_np(img_np, sigma=sharpen_sigma,
                                 strength=sharpen_strength)
    return torch.from_numpy(img_np.transpose(2, 0, 1)).float().to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    parser.add_argument("--mode", type=str, choices=["zero_shot", "adapt"], default="zero_shot")
    parser.add_argument("--disable_adarenet", action="store_true", help="Bypass AdaReNet (delta=0) for debugging")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--compute_lpips", action="store_true", help="Compute LPIPS metric (requires lpips package)")
    parser.add_argument("--no_color_correct", action="store_true", help="Disable gray-world color correction")
    parser.add_argument("--color_strength", type=float, default=0.3, help="Color correction strength (0=off, 1=full, default=0.3)")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (4x flip ensemble)")
    # Post-processing enhancement pipeline
    parser.add_argument("--enhance", action="store_true",
                        help="Enable post-processing pipeline (bilateral + contrast + sharpen)")
    parser.add_argument("--smooth_illum", type=float, default=0,
                        help="Gaussian sigma for L_T smoothing (0=off, recommended: 3.0)")
    parser.add_argument("--bilateral_sc", type=float, default=100,
                        help="Bilateral filter sigmaColor (higher=more smoothing, default=100)")
    parser.add_argument("--contrast_pct", type=float, default=1.0,
                        help="Contrast stretch clip percentile (0-5)")
    parser.add_argument("--sharpen_strength", type=float, default=0.3,
                        help="Unsharp mask strength (0-1)")
    # Advanced denoising methods
    parser.add_argument("--denoise_method", type=str, default="bilateral",
                        choices=["bilateral", "guided", "nlmeans", "combo"],
                        help="Denoising method: bilateral|guided|nlmeans|combo (default: bilateral)")
    parser.add_argument("--guided_radius", type=int, default=8,
                        help="Guided filter radius (default 8, larger=smoother)")
    parser.add_argument("--guided_eps", type=float, default=0.01,
                        help="Guided filter regularization (default 0.01, smaller=sharper edges)")
    parser.add_argument("--nlm_h", type=float, default=10,
                        help="Non-local means strength (default 10, higher=more smoothing)")
    # Individual post-processing toggles (when --enhance is set)
    parser.add_argument("--no_bilateral", action="store_true", help="Disable bilateral denoising")
    parser.add_argument("--no_contrast", action="store_true", help="Disable contrast stretching")
    parser.add_argument("--no_sharpen", action="store_true", help="Disable unsharp mask")
    # Model version
    parser.add_argument("--model_version", type=str, default="v1", choices=["v1", "v2lite"],
                        help="AdaReNet version: v1 (original) or v2lite (improved)")
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
    
    # Select model version
    if args.model_version == "v2lite":
        adarenet = AdaReNetV2Lite(base_channels=cfg["model"]["adarenet_channels"])
        print(f"[Infer] Using AdaReNetV2Lite ({sum(p.numel() for p in adarenet.parameters()):,} params)")
    else:
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
            
            # --- Core inference path ---
            use_smooth = args.smooth_illum > 0

            def _forward_single(inp):
                """Run single forward pass (standard or smooth-illum)."""
                if use_smooth:
                    return smooth_illumination_forward(model, inp, args.smooth_illum)
                out_ = model(inp)
                return out_["I_hat"][0]

            if args.disable_adarenet:
                l_t, l_e = model.compute_illumination(low)
                p_ref = model.compute_pref(low, l_t)
                i_hat = torch.clamp(p_ref * l_e, 0.0, 1.0)[0]
            elif args.tta:
                # Test-Time Augmentation: 4-fold flip ensemble
                augments = [
                    (lambda x: x, lambda x: x),
                    (lambda x: torch.flip(x, [3]), lambda x: torch.flip(x, [3])),
                    (lambda x: torch.flip(x, [2]), lambda x: torch.flip(x, [2])),
                    (lambda x: torch.flip(x, [2, 3]), lambda x: torch.flip(x, [2, 3])),
                ]
                i_hat_sum = torch.zeros_like(low[0])
                for fwd_fn, inv_fn in augments:
                    aug_i = _forward_single(fwd_fn(low))
                    i_hat_sum = i_hat_sum + inv_fn(aug_i.unsqueeze(0))[0]
                i_hat = i_hat_sum / len(augments)
            else:
                i_hat = _forward_single(low)

            # --- Color correction ---
            if not args.no_color_correct and args.color_strength > 0:
                i_hat = gray_world_correction(i_hat, strength=args.color_strength)

            # --- Post-processing pipeline ---
            if args.enhance:
                i_hat = postprocess_tensor(
                    i_hat,
                    bilateral=not args.no_bilateral,
                    contrast=not args.no_contrast,
                    sharpen=not args.no_sharpen,
                    bilateral_sc=args.bilateral_sc,
                    contrast_pct=args.contrast_pct,
                    sharpen_strength=args.sharpen_strength,
                    denoise_method=args.denoise_method,
                    guided_radius=args.guided_radius,
                    guided_eps=args.guided_eps,
                    nlm_h=args.nlm_h,
                )

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
