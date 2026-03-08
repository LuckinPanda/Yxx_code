#!/usr/bin/env python
"""
评估预生成的增强结果
Evaluate pre-generated enhanced images
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import compute_all_metrics, LPIPSWrapper


def load_image(path: str) -> torch.Tensor:
    """Load image as tensor [C, H, W] in [0, 1]."""
    img = Image.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    return img_tensor


def get_image_pairs(enhanced_dir: str, gt_dir: str) -> List[tuple]:
    """Get matched pairs of enhanced and GT images."""
    enhanced_dir = Path(enhanced_dir)
    gt_dir = Path(gt_dir)
    
    pairs = []
    for enh_path in sorted(enhanced_dir.glob("*.png")):
        if enh_path.is_dir():
            continue
        
        # Try to find matching GT image
        gt_path = gt_dir / enh_path.name
        if not gt_path.exists():
            for ext in ['.png', '.jpg', '.PNG', '.JPG']:
                alt_path = gt_dir / (enh_path.stem + ext)
                if alt_path.exists():
                    gt_path = alt_path
                    break
        
        if gt_path.exists():
            pairs.append((str(enh_path), str(gt_path)))
        else:
            print(f"[WARN] No GT found for {enh_path.name}")
    
    return pairs


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


def evaluate_method(enhanced_dir: str, gt_dir: str, method_name: str, 
                    device: torch.device, lpips_model: LPIPSWrapper = None,
                    compute_lpips: bool = True) -> Dict:
    """Evaluate a single method."""
    pairs = get_image_pairs(enhanced_dir, gt_dir)
    
    if not pairs:
        print(f"[ERROR] No image pairs found for {method_name}")
        return {'statistics': {}, 'per_image': []}
    
    metrics_list = []
    
    for enh_path, gt_path in tqdm(pairs, desc=method_name, ncols=80):
        try:
            enhanced = load_image(enh_path).to(device)
            gt = load_image(gt_path).to(device)
            
            metrics = compute_all_metrics(
                enhanced, gt,
                lpips_model=lpips_model,
                compute_lpips=compute_lpips
            )
            metrics['image_name'] = Path(enh_path).name
            metrics_list.append(metrics)
            
        except Exception as e:
            print(f"[ERROR] Failed {enh_path}: {e}")
    
    stats = compute_statistics(metrics_list)
    return {'statistics': stats, 'per_image': metrics_list}


def main():
    parser = argparse.ArgumentParser(description='Evaluate enhanced images')
    parser.add_argument('--enhanced_dir', type=str, required=True,
                        help='Directory containing enhanced images')
    parser.add_argument('--gt_dir', type=str, 
                        default='/home/yannayanna/datasets/LOL/eval15/high',
                        help='Directory containing ground truth images')
    parser.add_argument('--method_name', type=str, default='Unknown',
                        help='Name of the method')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as enhanced_dir)')
    parser.add_argument('--no_lpips', action='store_true',
                        help='Skip LPIPS computation')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Evaluate] Using device: {device}")
    
    lpips_model = None
    if not args.no_lpips:
        try:
            lpips_model = LPIPSWrapper(device)
        except Exception as e:
            print(f"[WARN] Failed to load LPIPS: {e}")
    
    results = evaluate_method(
        args.enhanced_dir, args.gt_dir, args.method_name,
        device, lpips_model, not args.no_lpips
    )
    
    stats = results['statistics']
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results for: {args.method_name}")
    print(f"{'='*60}")
    if 'psnr' in stats:
        print(f"  PSNR:  {stats['psnr']['mean']:.4f} ± {stats['psnr']['std']:.4f}")
    if 'ssim' in stats:
        print(f"  SSIM:  {stats['ssim']['mean']:.4f} ± {stats['ssim']['std']:.4f}")
    if 'mae' in stats:
        print(f"  MAE:   {stats['mae']['mean']:.4f} ± {stats['mae']['std']:.4f}")
    if 'lpips' in stats:
        print(f"  LPIPS: {stats['lpips']['mean']:.4f} ± {stats['lpips']['std']:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.enhanced_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-image CSV
    csv_path = output_dir / "metrics_per_image.csv"
    if results['per_image']:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results['per_image'][0].keys())
            writer.writeheader()
            writer.writerows(results['per_image'])
        print(f"Per-image metrics saved to: {csv_path}")
    
    # Save summary JSON
    json_path = output_dir / "metrics_summary.json"
    summary = {
        'method': args.method_name,
        'timestamp': datetime.now().isoformat(),
        'gt_dir': args.gt_dir,
        'statistics': stats
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {json_path}")


if __name__ == "__main__":
    main()
