#!/usr/bin/env python
"""
汇总所有方法的评估结果
Summarize all method results
"""

import json
from datetime import datetime
from pathlib import Path

# 收集所有结果
results = {
    # Traditional Methods
    "HE": {
        "PSNR": 14.2290,
        "SSIM": 0.4275,
        "MAE": 0.1726,
        "LPIPS": 0.5920,
        "category": "Traditional"
    },
    "CLAHE": {
        "PSNR": 9.1323,
        "SSIM": 0.3587,
        "MAE": 0.3372,
        "LPIPS": 0.5029,
        "category": "Traditional"
    },
    "Gamma": {
        "PSNR": 14.0258,
        "SSIM": 0.6122,
        "MAE": 0.1980,
        "LPIPS": 0.3480,
        "category": "Traditional"
    },
    "MSR": {
        "PSNR": 7.9273,
        "SSIM": 0.4872,
        "MAE": 0.3829,
        "LPIPS": 0.6677,
        "category": "Traditional"
    },
    # Deep Learning Methods
    "Zero-DCE": {
        "PSNR": 14.7971,
        "SSIM": 0.5598,
        "MAE": 0.1860,
        "LPIPS": 0.3352,
        "category": "Deep"
    },
    "SCI": {
        "PSNR": 14.7840,
        "SSIM": 0.5230,
        "MAE": 0.1912,
        "LPIPS": 0.3393,
        "category": "Deep"
    },
    "EnlightenGAN": {
        "PSNR": 17.8934,
        "SSIM": 0.7114,
        "MAE": 0.1212,
        "LPIPS": 0.3014,
        "category": "Deep"
    },
    "Ours (Retinex-AdaReNet)": {
        "PSNR": 17.9801,
        "SSIM": 0.5025,
        "MAE": 0.1199,
        "LPIPS": 0.3712,
        "category": "Deep"
    },
    "URetinex-Net++": {
        "PSNR": 20.7897,
        "SSIM": 0.8236,
        "MAE": 0.0937,
        "LPIPS": 0.1163,
        "category": "Deep"
    }
}

# 打印对比表格
print("=" * 100)
print("LOL eval15 数据集 低光增强方法对比结果")
print("=" * 100)
print(f"{'Method':<30} {'PSNR↑':<12} {'SSIM↑':<12} {'MAE↓':<12} {'LPIPS↓':<12}")
print("-" * 100)

# 按类别分组打印
print("\n--- Traditional Methods ---")
for method, metrics in results.items():
    if metrics.get('category') == 'Traditional':
        print(f"{method:<30} {metrics['PSNR']:<12.4f} {metrics['SSIM']:<12.4f} {metrics['MAE']:<12.4f} {metrics['LPIPS']:<12.4f}")

print("\n--- Deep Learning Methods ---")
for method, metrics in results.items():
    if metrics.get('category') == 'Deep':
        print(f"{method:<30} {metrics['PSNR']:<12.4f} {metrics['SSIM']:<12.4f} {metrics['MAE']:<12.4f} {metrics['LPIPS']:<12.4f}")

print("-" * 100)
print("↑ 越高越好, ↓ 越低越好")
print()

# 按 PSNR 排名
print("\n=== PSNR 排名 ===")
sorted_by_psnr = sorted(results.items(), key=lambda x: x[1]['PSNR'], reverse=True)
for i, (method, metrics) in enumerate(sorted_by_psnr, 1):
    print(f"{i}. {method}: {metrics['PSNR']:.4f}")

# 按 SSIM 排名
print("\n=== SSIM 排名 ===")
sorted_by_ssim = sorted(results.items(), key=lambda x: x[1]['SSIM'], reverse=True)
for i, (method, metrics) in enumerate(sorted_by_ssim, 1):
    print(f"{i}. {method}: {metrics['SSIM']:.4f}")

# 按 LPIPS 排名
print("\n=== LPIPS 排名 (越低越好) ===")
sorted_by_lpips = sorted(results.items(), key=lambda x: x[1]['LPIPS'])
for i, (method, metrics) in enumerate(sorted_by_lpips, 1):
    print(f"{i}. {method}: {metrics['LPIPS']:.4f}")

# 生成 LaTeX 表格
print("\n" + "=" * 100)
print("LaTeX 表格:")
print("=" * 100)
latex = r"""
\begin{table}[h]
\centering
\caption{Low-Light Image Enhancement Results on LOL eval15 Dataset}
\label{tab:lol_benchmark}
\begin{tabular}{lcccc}
\toprule
Method & PSNR$\uparrow$ & SSIM$\uparrow$ & MAE$\downarrow$ & LPIPS$\downarrow$ \\
\midrule
\multicolumn{5}{l}{\textit{Traditional Methods}} \\
HE & 14.23 & 0.4275 & 0.1726 & 0.5920 \\
CLAHE & 9.13 & 0.3587 & 0.3372 & 0.5029 \\
Gamma Correction & 14.03 & 0.6122 & 0.1980 & 0.3480 \\
MSR & 7.93 & 0.4872 & 0.3829 & 0.6677 \\
\midrule
\multicolumn{5}{l}{\textit{Deep Learning Methods}} \\
Zero-DCE & 14.80 & 0.5598 & 0.1860 & 0.3352 \\
SCI & 14.78 & 0.5230 & 0.1912 & 0.3393 \\
EnlightenGAN & 17.89 & 0.7114 & 0.1212 & 0.3014 \\
Ours (Retinex-AdaReNet) & 17.98 & 0.5025 & 0.1199 & 0.3712 \\
URetinex-Net++ & \textbf{20.79} & \textbf{0.8236} & \textbf{0.0937} & \textbf{0.1163} \\
\bottomrule
\end{tabular}
\end{table}
"""
print(latex)

# 保存结果到 JSON
output_path = Path(__file__).parent / "benchmark_results.json"
with open(output_path, 'w') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "dataset": "LOL eval15",
        "num_images": 15,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != 'category'} for k, v in results.items()}
    }, f, indent=2)
print(f"\n结果已保存到: {output_path}")
