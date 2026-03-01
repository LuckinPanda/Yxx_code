#!/usr/bin/env python3
"""
Multi-seed stability evaluation script.
Runs inference with multiple random seeds and aggregates results.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_inference(mode: str, config: str, seed: int, compute_lpips: bool = False) -> Path:
    """Run inference with specified seed and return output directory."""
    cmd = [
        sys.executable,
        "infer.py",
        "--config", config,
        "--mode", mode,
        "--seed", str(seed),
    ]
    if compute_lpips:
        cmd.append("--compute_lpips")
    
    print(f"\n{'='*60}")
    print(f"Running inference: mode={mode}, seed={seed}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        raise RuntimeError(f"Inference failed with seed {seed}")
    
    # Find the most recent output directory for this mode and seed
    output_base = Path("outputs")
    matching_dirs = sorted(output_base.glob(f"{mode}_*_seed{seed}"))
    if not matching_dirs:
        raise RuntimeError(f"Could not find output directory for seed {seed}")
    
    return matching_dirs[-1]


def aggregate_results(output_dirs: list[Path], output_path: Path) -> None:
    """Aggregate results from multiple seeds."""
    all_data = []
    
    for out_dir in output_dirs:
        # Extract seed from directory name
        dir_name = out_dir.name
        seed = int(dir_name.split("_seed")[-1])
        
        # Load per-image metrics
        csv_path = out_dir / "metrics_per_image.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping")
            continue
        
        df = pd.read_csv(csv_path)
        df['seed'] = seed
        all_data.append(df)
    
    if not all_data:
        raise RuntimeError("No valid data found")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Compute per-image statistics across seeds
    per_image_stats = combined_df.groupby('image_name').agg({
        'psnr': ['mean', 'std', 'min', 'max'],
        'ssim': ['mean', 'std', 'min', 'max'],
    })
    
    if 'lpips' in combined_df.columns:
        lpips_stats = combined_df.groupby('image_name')['lpips'].agg(['mean', 'std', 'min', 'max'])
        per_image_stats = pd.concat([per_image_stats, lpips_stats], axis=1)
    
    per_image_stats = per_image_stats.reset_index()
    per_image_stats.columns = ['_'.join(col).strip('_') for col in per_image_stats.columns.values]
    
    # Save per-image cross-seed statistics
    per_image_csv = output_path / "per_image_cross_seed_stats.csv"
    per_image_stats.to_csv(per_image_csv, index=False)
    print(f"\n[MultiSeed] Saved per-image cross-seed statistics to: {per_image_csv}")
    
    # Compute global statistics
    global_stats = {
        'num_seeds': len(output_dirs),
        'num_images': len(per_image_stats),
        'psnr': {
            'mean_across_seeds': combined_df['psnr'].mean(),
            'std_across_seeds': combined_df['psnr'].std(),
            'mean_of_stds': per_image_stats['psnr_std'].mean(),
            'max_std': per_image_stats['psnr_std'].max(),
            'image_with_max_std': per_image_stats.loc[per_image_stats['psnr_std'].idxmax(), 'image_name'],
        },
        'ssim': {
            'mean_across_seeds': combined_df['ssim'].mean(),
            'std_across_seeds': combined_df['ssim'].std(),
            'mean_of_stds': per_image_stats['ssim_std'].mean(),
            'max_std': per_image_stats['ssim_std'].max(),
            'image_with_max_std': per_image_stats.loc[per_image_stats['ssim_std'].idxmax(), 'image_name'],
        },
    }
    
    if 'lpips' in combined_df.columns:
        global_stats['lpips'] = {
            'mean_across_seeds': combined_df['lpips'].mean(),
            'std_across_seeds': combined_df['lpips'].std(),
            'mean_of_stds': per_image_stats['lpips_std'].mean(),
            'max_std': per_image_stats['lpips_std'].max(),
            'image_with_max_std': per_image_stats.loc[per_image_stats['lpips_std'].idxmax(), 'image_name'],
        }
    
    # Compute gain distribution (relative to baseline)
    # For each image, compute variance across seeds
    gain_variance = per_image_stats[['image_name', 'psnr_std', 'ssim_std']].copy()
    gain_variance['psnr_coefficient_of_variation'] = (
        per_image_stats['psnr_std'] / per_image_stats['psnr_mean']
    )
    gain_variance['ssim_coefficient_of_variation'] = (
        per_image_stats['ssim_std'] / per_image_stats['ssim_mean']
    )
    
    gain_csv = output_path / "per_image_gain_variance.csv"
    gain_variance.to_csv(gain_csv, index=False)
    print(f"[MultiSeed] Saved per-image gain variance to: {gain_csv}")
    
    # Save global summary
    summary_json = output_path / "multi_seed_summary.json"
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(global_stats, f, indent=2, ensure_ascii=False)
    print(f"[MultiSeed] Saved global summary to: {summary_json}")
    
    # Print summary
    print("\n" + "="*60)
    print("MULTI-SEED STABILITY ANALYSIS")
    print("="*60)
    print(f"Number of seeds: {global_stats['num_seeds']}")
    print(f"Number of images: {global_stats['num_images']}")
    
    print(f"\nPSNR Stability:")
    print(f"  Mean across all: {global_stats['psnr']['mean_across_seeds']:.4f}")
    print(f"  Std across all:  {global_stats['psnr']['std_across_seeds']:.4f}")
    print(f"  Avg per-image std: {global_stats['psnr']['mean_of_stds']:.4f}")
    print(f"  Max per-image std: {global_stats['psnr']['max_std']:.4f}")
    print(f"    (image: {global_stats['psnr']['image_with_max_std']})")
    
    print(f"\nSSIM Stability:")
    print(f"  Mean across all: {global_stats['ssim']['mean_across_seeds']:.4f}")
    print(f"  Std across all:  {global_stats['ssim']['std_across_seeds']:.4f}")
    print(f"  Avg per-image std: {global_stats['ssim']['mean_of_stds']:.4f}")
    print(f"  Max per-image std: {global_stats['ssim']['max_std']:.4f}")
    print(f"    (image: {global_stats['ssim']['image_with_max_std']})")
    
    if 'lpips' in global_stats:
        print(f"\nLPIPS Stability:")
        print(f"  Mean across all: {global_stats['lpips']['mean_across_seeds']:.4f}")
        print(f"  Std across all:  {global_stats['lpips']['std_across_seeds']:.4f}")
        print(f"  Avg per-image std: {global_stats['lpips']['mean_of_stds']:.4f}")
        print(f"  Max per-image std: {global_stats['lpips']['max_std']:.4f}")
        print(f"    (image: {global_stats['lpips']['image_with_max_std']})")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed stability evaluation")
    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    parser.add_argument("--mode", type=str, choices=["zero_shot", "adapt"], default="zero_shot")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024],
                        help="List of random seeds to test")
    parser.add_argument("--compute_lpips", action="store_true", help="Compute LPIPS metric")
    args = parser.parse_args()
    
    print(f"Starting multi-seed evaluation with {len(args.seeds)} seeds: {args.seeds}")
    
    output_dirs = []
    for seed in args.seeds:
        try:
            out_dir = run_inference(args.mode, args.config, seed, args.compute_lpips)
            output_dirs.append(out_dir)
        except Exception as e:
            print(f"ERROR: Failed for seed {seed}: {e}")
            continue
    
    if not output_dirs:
        print("ERROR: No successful runs, exiting")
        return
    
    # Aggregate results
    aggregated_output = Path("outputs") / f"multi_seed_{args.mode}"
    aggregated_output.mkdir(parents=True, exist_ok=True)
    
    aggregate_results(output_dirs, aggregated_output)
    print(f"\n[MultiSeed] All results saved to: {aggregated_output}")


if __name__ == "__main__":
    main()
