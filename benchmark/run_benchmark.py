#!/usr/bin/env python
"""
统一基准评估脚本
Unified benchmark evaluation script

用法:
    python benchmark/run_benchmark.py --methods all
    python benchmark/run_benchmark.py --methods HE CLAHE Ours
    python benchmark/run_benchmark.py --methods Ours URetinex++ --no_lpips
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import compute_all_metrics, LPIPSWrapper
from benchmark.traditional_methods import get_traditional_methods
from benchmark.deep_methods import ZeroDCE, URetinexNetPP, RetinexAdaReNetWrapper


def load_image(path: str) -> torch.Tensor:
    """Load image as tensor [C, H, W] in [0, 1]."""
    img = Image.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    return img_tensor


def save_image(tensor: torch.Tensor, path: str):
    """Save tensor [C, H, W] as image."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    img_np = tensor.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_np).save(path)


def get_image_pairs(low_dir: str, high_dir: str) -> List[tuple]:
    """Get matched pairs of low and high images."""
    low_dir = Path(low_dir)
    high_dir = Path(high_dir)
    
    pairs = []
    for low_path in sorted(low_dir.glob("*.png")):
        if low_path.is_dir():
            continue
        # Try to find matching high image
        high_path = high_dir / low_path.name
        if not high_path.exists():
            # Try without extension match
            for ext in ['.png', '.jpg', '.PNG', '.JPG']:
                alt_path = high_dir / (low_path.stem + ext)
                if alt_path.exists():
                    high_path = alt_path
                    break
        
        if high_path.exists():
            pairs.append((str(low_path), str(high_path)))
        else:
            print(f"[WARN] No GT found for {low_path.name}, skipping")
    
    return pairs


def get_all_methods(device: torch.device, args) -> Dict:
    """Get all available enhancement methods."""
    methods = {}
    
    # Traditional methods
    traditional = get_traditional_methods()
    for name, method in traditional.items():
        methods[name] = method
    
    # Deep learning methods
    try:
        methods["Zero-DCE"] = ZeroDCE(
            ckpt_path=getattr(args, 'zerodce_ckpt', None),
            device=device
        )
    except Exception as e:
        print(f"[WARN] Failed to load Zero-DCE: {e}")
    
    try:
        methods["URetinex++"] = URetinexNetPP(
            project_path="/home/yannayanna/projects/URetinex-Net-PLUS",
            ratio=5.0,
            device=device
        )
    except Exception as e:
        print(f"[WARN] Failed to load URetinex++: {e}")
    
    try:
        methods["Ours"] = RetinexAdaReNetWrapper(
            project_path=str(PROJECT_ROOT),
            mode="zero_shot",
            device=device,
            color_correct=True,
            color_strength=0.3,
        )
    except Exception as e:
        print(f"[WARN] Failed to load Ours: {e}")
    
    return methods


def compute_statistics(metrics_list: List[Dict]) -> Dict:
    """Compute statistics from per-image metrics."""
    if not metrics_list:
        return {}
    
    stats = {'count': len(metrics_list)}
    
    for key in ['psnr', 'ssim', 'mae', 'lpips']:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
    
    return stats


def run_benchmark(args):
    """Run benchmark evaluation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Benchmark] Using device: {device}")
    
    # Get image pairs
    pairs = get_image_pairs(args.low_dir, args.high_dir)
    print(f"[Benchmark] Found {len(pairs)} image pairs")
    
    if len(pairs) == 0:
        print("[ERROR] No image pairs found!")
        return
    
    # Get methods
    all_methods = get_all_methods(device, args)
    
    if args.methods == ['all']:
        methods_to_run = list(all_methods.keys())
    else:
        methods_to_run = args.methods
    
    print(f"[Benchmark] Methods to evaluate: {methods_to_run}")
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize LPIPS if needed
    lpips_model = None
    if not args.no_lpips:
        try:
            lpips_model = LPIPSWrapper(device)
        except Exception as e:
            print(f"[WARN] Failed to load LPIPS: {e}")
    
    # Results storage
    all_results = {}
    
    for method_name in methods_to_run:
        if method_name not in all_methods:
            print(f"[WARN] Method '{method_name}' not available, skipping")
            continue
        
        method = all_methods[method_name]
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*60}")
        
        # Create method output directory
        method_dir = output_dir / method_name.replace(' ', '_').replace('+', 'plus')
        method_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_list = []
        
        for low_path, high_path in tqdm(pairs, desc=method_name, ncols=80):
            try:
                # Load images
                low_img = load_image(low_path).to(device)
                high_img = load_image(high_path).to(device)
                
                # Enhance
                if hasattr(method, 'enhance_tensor'):
                    enhanced = method.enhance_tensor(low_img)
                else:
                    enhanced = method.enhance(low_img)
                
                enhanced = enhanced.to(device)
                
                # Compute metrics
                metrics = compute_all_metrics(
                    enhanced, high_img,
                    lpips_model=lpips_model,
                    compute_lpips=(not args.no_lpips and lpips_model is not None)
                )
                metrics['image_name'] = Path(low_path).name
                metrics_list.append(metrics)
                
                # Save enhanced image
                if args.save_images:
                    save_path = method_dir / Path(low_path).name
                    save_image(enhanced, str(save_path))
                
            except Exception as e:
                print(f"[ERROR] Failed to process {low_path}: {e}")
                continue
        
        # Compute statistics
        stats = compute_statistics(metrics_list)
        all_results[method_name] = {
            'statistics': stats,
            'per_image': metrics_list
        }
        
        # Print summary
        if 'psnr' in stats:
            print(f"  PSNR: {stats['psnr']['mean']:.4f} ± {stats['psnr']['std']:.4f}")
        if 'ssim' in stats:
            print(f"  SSIM: {stats['ssim']['mean']:.4f} ± {stats['ssim']['std']:.4f}")
        if 'mae' in stats:
            print(f"  MAE:  {stats['mae']['mean']:.4f} ± {stats['mae']['std']:.4f}")
        if 'lpips' in stats:
            print(f"  LPIPS: {stats['lpips']['mean']:.4f} ± {stats['lpips']['std']:.4f}")
        
        # Save per-method metrics
        csv_path = method_dir / "metrics_per_image.csv"
        if metrics_list:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics_list[0].keys())
                writer.writeheader()
                writer.writerows(metrics_list)
    
    # Save combined results
    summary_path = output_dir / "benchmark_summary.json"
    summary_data = {
        'metadata': {
            'timestamp': timestamp,
            'low_dir': args.low_dir,
            'high_dir': args.high_dir,
            'num_images': len(pairs),
            'methods': methods_to_run,
        },
        'results': {k: v['statistics'] for k, v in all_results.items()}
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS COMPARISON")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'PSNR↑':<12} {'SSIM↑':<12} {'MAE↓':<12} {'LPIPS↓':<12}")
    print(f"{'-'*80}")
    
    for method_name in methods_to_run:
        if method_name not in all_results:
            continue
        stats = all_results[method_name]['statistics']
        psnr = f"{stats['psnr']['mean']:.4f}" if 'psnr' in stats else "N/A"
        ssim = f"{stats['ssim']['mean']:.4f}" if 'ssim' in stats else "N/A"
        mae = f"{stats['mae']['mean']:.4f}" if 'mae' in stats else "N/A"
        lpips_val = f"{stats['lpips']['mean']:.4f}" if 'lpips' in stats else "N/A"
        print(f"{method_name:<15} {psnr:<12} {ssim:<12} {mae:<12} {lpips_val:<12}")
    
    print(f"{'-'*80}")
    print(f"Results saved to: {output_dir}")
    
    # Generate LaTeX table
    latex_path = output_dir / "results_table.tex"
    with open(latex_path, 'w') as f:
        f.write("% Auto-generated benchmark results table\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Low-Light Enhancement Benchmark on LOL Dataset}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Method & PSNR$\\uparrow$ & SSIM$\\uparrow$ & MAE$\\downarrow$ & LPIPS$\\downarrow$ \\\\\n")
        f.write("\\midrule\n")
        
        for method_name in methods_to_run:
            if method_name not in all_results:
                continue
            stats = all_results[method_name]['statistics']
            psnr = f"{stats['psnr']['mean']:.2f}" if 'psnr' in stats else "-"
            ssim = f"{stats['ssim']['mean']:.4f}" if 'ssim' in stats else "-"
            mae = f"{stats['mae']['mean']:.4f}" if 'mae' in stats else "-"
            lpips_val = f"{stats['lpips']['mean']:.4f}" if 'lpips' in stats else "-"
            
            # Escape special characters
            safe_name = method_name.replace('++', '$++$').replace('_', '\\_')
            f.write(f"{safe_name} & {psnr} & {ssim} & {mae} & {lpips_val} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:benchmark}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {latex_path}")


def main():
    parser = argparse.ArgumentParser(description='Unified Low-Light Enhancement Benchmark')
    parser.add_argument('--low_dir', type=str, 
                        default='/home/yannayanna/datasets/LOL/eval15/low',
                        help='Directory containing low-light test images')
    parser.add_argument('--high_dir', type=str,
                        default='/home/yannayanna/datasets/LOL/eval15/high',
                        help='Directory containing ground truth images')
    parser.add_argument('--output_dir', type=str,
                        default='/home/yannayanna/projects/retinex_adarenet/outputs/benchmark',
                        help='Output directory for results')
    parser.add_argument('--methods', nargs='+', default=['all'],
                        help='Methods to evaluate: all, HE, CLAHE, Gamma, MSR, Zero-DCE, URetinex++, Ours')
    parser.add_argument('--no_lpips', action='store_true',
                        help='Skip LPIPS computation')
    parser.add_argument('--save_images', action='store_true', default=True,
                        help='Save enhanced images')
    parser.add_argument('--no_save_images', action='store_false', dest='save_images',
                        help='Do not save enhanced images')
    parser.add_argument('--zerodce_ckpt', type=str, default=None,
                        help='Path to Zero-DCE checkpoint')
    
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
