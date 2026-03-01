"""
消融实验: 仅输出提亮步骤结果 (不经过去噪)

功能:
  1. 加载 IlluminationNet 模型
  2. 对所有低光图片执行 Retinex 分解 + 光照增强
  3. 输出 I_hat_nodenoise = clamp(P_ref * L_e, 0, 1)  （仅提亮，未去噪）
  4. 同时保存中间结果 (L_T, L_e, P_ref) 供分析
  5. 若有 ground truth，计算 PSNR/SSIM 指标

用法:
  cd /home/yannayanna/projects/retinex_adarenet
  python infer_illum_only.py --config configs/infer.yaml
  python infer_illum_only.py --config configs/infer.yaml --save_intermediate
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LowLightDataset
from src.models.illumination import IlluminationNet
from src.utils.config import load_config
from src.utils.image import save_image


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)


def compute_psnr(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def compute_ssim(pred, target, window_size=11, sigma=1.5, data_range=1.0):
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    _, channels, _, _ = pred.shape
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    window = window.repeat(channels, 1, 1, 1)
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    s2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    s12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1_sq + s2_sq + c2))
    return ssim_map.mean()


def main():
    parser = argparse.ArgumentParser(description="消融实验: 仅输出 Retinex 提亮结果 (不经过去噪)")
    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_intermediate", action="store_true",
                        help="同时保存中间结果 L_T, L_e, P_ref")
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Constants
    omega = float(cfg["constants"]["omega"])
    tau = float(cfg["constants"]["tau"])
    eps = float(cfg["constants"]["eps"])
    illum_adjust_mode = cfg["constants"].get("illum_adjust_mode", "gamma")

    # Dataset
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

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=data_cfg["num_workers"], pin_memory=True)

    # Load illumination model only
    illum = IlluminationNet(base_channels=cfg["model"]["illumination_channels"]).to(device)
    illum_ckpt = cfg["ckpt"]["illum_ckpt_path"]
    illum.load_state_dict(torch.load(illum_ckpt, map_location=device))
    illum.eval()
    print(f"[IllumOnly] Loaded illumination model from: {illum_ckpt}")
    print(f"[IllumOnly] omega={omega}, tau={tau}, eps={eps}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = Path(data_cfg["output_dir"])
    out_dir = base_out_dir / f"illum_only_{timestamp}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_intermediate:
        (out_dir / "L_T").mkdir(exist_ok=True)
        (out_dir / "L_e").mkdir(exist_ok=True)
        (out_dir / "P_ref").mkdir(exist_ok=True)

    print(f"[IllumOnly] Output directory: {out_dir}")
    print(f"[IllumOnly] 提亮图将保存到: {out_dir}  (可直接用于 AdaReNet 去噪测试)")

    metrics_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Illumination Only", ncols=80):
            low = batch["low"].to(device, non_blocking=True)
            name = batch["name"][0]

            # Step 1: Illumination estimation
            l_t = illum(low)  # [B, 1, H, W]
            if illum_adjust_mode == "gamma":
                l_e = torch.clamp(l_t.pow(1.0 / omega), 0.0, 1.0)
            else:
                l_e = torch.clamp(l_t * omega, 0.0, 1.0)

            # Step 2: Reflectance proxy (without denoising)
            l_tilde = torch.clamp(l_t, min=tau)
            p_ref = low / (l_tilde + eps)

            # Step 3: Reconstruct (no denoising, just illumination enhancement)
            i_hat = torch.clamp(p_ref * l_e, 0.0, 1.0)  # [B, 3, H, W]

            # Save main output
            save_image(i_hat[0], str(out_dir / name))

            # Save intermediate results
            if args.save_intermediate:
                # L_T: single channel → repeat to 3ch for visualization
                l_t_vis = l_t[0].repeat(3, 1, 1)
                save_image(l_t_vis, str(out_dir / "L_T" / name))

                l_e_vis = l_e[0].repeat(3, 1, 1)
                save_image(l_e_vis, str(out_dir / "L_e" / name))

                # P_ref: clamp for visualization (may be > 1)
                p_ref_vis = torch.clamp(p_ref[0], 0.0, 1.0)
                save_image(p_ref_vis, str(out_dir / "P_ref" / name))

            # Compute metrics if ground truth available
            if "high" in batch:
                high = batch["high"].to(device, non_blocking=True)
                pred = i_hat
                psnr_val = compute_psnr(pred, high).item()
                ssim_val = compute_ssim(pred, high).item()
                metrics_list.append({
                    "image_name": name,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                })

    print(f"\n[IllumOnly] Saved {len(loader)} brightened images to: {out_dir}")

    # Save and display metrics
    if metrics_list:
        # Per-image CSV
        csv_path = out_dir / "metrics_per_image.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_list[0].keys())
            writer.writeheader()
            writer.writerows(metrics_list)

        psnrs = [m["psnr"] for m in metrics_list]
        ssims = [m["ssim"] for m in metrics_list]

        stats = {
            "experiment": "illumination_only (no denoising)",
            "count": len(metrics_list),
            "omega": omega,
            "tau": tau,
            "eps": eps,
            "psnr_mean": float(np.mean(psnrs)),
            "psnr_std": float(np.std(psnrs)),
            "ssim_mean": float(np.mean(ssims)),
            "ssim_std": float(np.std(ssims)),
        }

        json_path = out_dir / "metrics_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("消融实验: 仅提亮 (无去噪) 评估结果")
        print("=" * 60)
        print(f"图片数量: {stats['count']}")
        print(f"PSNR: {stats['psnr_mean']:.4f} ± {stats['psnr_std']:.4f} dB")
        print(f"SSIM: {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}")
        print("=" * 60)
        print(f"\n下一步: 将 {out_dir} 中的图片送入 AdaReNet 去噪:")
        print(f"  cd /home/yannayanna/projects/AdaReNet")
        print(f"  python test_ablation.py \\")
        print(f"    --noisy-dir {out_dir} \\")
        print(f"    --clean-dir {test_high_dir or '<ground_truth_dir>'} \\")
        print(f"    --load-ckpt <your_adarenet_checkpoint.pt> \\")
        print(f"    --cuda")
        print()


if __name__ == "__main__":
    main()
