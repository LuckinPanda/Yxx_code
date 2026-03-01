# 评估指南

## 功能概述

已实现的评估功能包括：

1. **时间戳输出目录** - 每次推理自动创建带时间戳的新目录，避免覆盖
2. **逐图指标记录** - 每张图片的 PSNR/SSIM/LPIPS 指标单独保存
3. **统计分析** - 包含均值、标准差、中位数、最大/最小值、最差10%平均值
4. **多种子稳定性评估** - 测试不同随机种子下的模型稳定性
5. **逐图增益分布** - 分析每张图片在不同种子下的指标方差

## 使用方法

### 1. 单次推理评估

```bash
# 基础评估（PSNR + SSIM），默认开启灰度世界颜色校正
python infer.py --mode zero_shot --config configs/infer.yaml --seed 42

# 添加 LPIPS 评估（需要网络下载预训练模型）
python infer.py --mode zero_shot --config configs/infer.yaml --seed 42 --compute_lpips

# 关闭颜色校正
python infer.py --mode zero_shot --config configs/infer.yaml --seed 42 --no_color_correct

# 调整颜色校正强度 (0=关闭, 1=全开, 默认0.8)
python infer.py --mode zero_shot --config configs/infer.yaml --seed 42 --color_strength 0.6

# adapt 模式评估 (deprecated)
python infer.py --mode adapt --config configs/infer.yaml --seed 123
```

**输出文件：**
- `outputs/{mode}_{timestamp}_seed{seed}/`
  - `*.png` - 增强后的图像
  - `metrics_per_image.csv` - 每张图片的详细指标
  - `metrics_summary.json` - 统计汇总（包含元数据）

**CSV 格式示例：**
```csv
image_name,psnr,ssim,lpips
low00690.png,9.0058,0.3532,0.1234
low00691.png,9.0024,0.3526,0.1456
...
```

### 2. 多种子稳定性评估

```bash
# 使用默认5个种子 [42, 123, 456, 789, 2024]
python eval_multi_seed.py --mode zero_shot --config configs/infer.yaml

# 自定义种子
python eval_multi_seed.py --mode zero_shot --seeds 42 123 456 --config configs/infer.yaml

# 包含 LPIPS 评估
python eval_multi_seed.py --mode adapt --seeds 42 123 --compute_lpips --config configs/infer.yaml
```

**输出文件：**
- `outputs/multi_seed_{mode}/`
  - `per_image_cross_seed_stats.csv` - 每张图片跨种子的统计（均值/标准差/最大/最小）
  - `per_image_gain_variance.csv` - 每张图片的方差和变异系数
  - `multi_seed_summary.json` - 全局稳定性统计

**Cross-seed stats CSV 格式：**
```csv
image_name,psnr_mean,psnr_std,psnr_min,psnr_max,ssim_mean,ssim_std,ssim_min,ssim_max
low00690.png,9.0058,0.0012,9.0045,9.0070,0.3532,0.0001,0.3531,0.3533
...
```

## 输出指标说明

### 单张图片指标
- **PSNR** (峰值信噪比): 越高越好，单位 dB
- **SSIM** (结构相似度): 0-1 之间，越高越好
- **LPIPS** (感知相似度): 越低越好（需要额外安装：`pip install lpips`）

### 统计指标
- **Mean**: 平均值
- **Std**: 标准差（衡量指标波动）
- **Median**: 中位数（不受极端值影响）
- **Min/Max**: 最小/最大值
- **Worst 10%**: 最差10%图片的平均值（识别困难样本）

### 稳定性指标（多种子）
- **Mean across all**: 所有种子所有图片的总平均值
- **Std across all**: 所有数据的总标准差
- **Avg per-image std**: 每张图片跨种子标准差的平均值（衡量模型稳定性）
- **Max per-image std**: 标准差最大的图片（识别不稳定样本）
- **Coefficient of variation**: 变异系数 = std/mean（相对波动）

## 典型应用场景

### 场景1：对比不同模型版本
```bash
# 测试版本A
python infer.py --mode zero_shot --config configs/infer.yaml --seed 42

# 测试版本B（修改配置或checkpoint后）
python infer.py --mode zero_shot --config configs/infer.yaml --seed 42

# 测试不同颜色校正强度
python infer.py --mode zero_shot --config configs/infer.yaml --seed 42 --color_strength 0.5
python infer.py --mode zero_shot --config configs/infer.yaml --seed 42 --color_strength 1.0
python infer.py --mode zero_shot --config configs/infer.yaml --seed 42 --no_color_correct

# 对比输出目录中的 metrics_summary.json
```

### 场景2：评估模型稳定性
```bash
# 运行多种子评估
python eval_multi_seed.py --mode zero_shot --seeds 42 123 456 789 2024

# 查看 outputs/multi_seed_zero_shot/multi_seed_summary.json
# 重点关注：
# - psnr.mean_of_stds: 平均波动幅度
# - psnr.max_std: 最不稳定图片的波动
```

### 场景3：找出困难样本
```bash
# 运行评估
python infer.py --mode zero_shot --config configs/infer.yaml --seed 42

# 打开 metrics_per_image.csv，按 psnr 升序排序
# 最差的10-20张图片即为困难样本
```

### 场景4：分析adapt相对zero_shot的提升
```bash
# 分别运行两种模式
python infer.py --mode zero_shot --seed 42 --config configs/infer.yaml
python infer.py --mode adapt --seed 42 --config configs/infer.yaml

# 对比两个 metrics_per_image.csv 文件
# 计算逐图增益：gain = adapt_psnr - zero_shot_psnr
```

## 注意事项

### LPIPS 使用
LPIPS 需要下载预训练的 AlexNet 模型（~200MB），首次使用时会自动下载。如果网络不稳定：
```bash
# 方案1：手动下载并放置到缓存目录
# 下载地址：https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
# 放置位置：~/.cache/torch/hub/checkpoints/

# 方案2：暂时跳过 LPIPS
python infer.py --mode zero_shot --config configs/infer.yaml  # 不加 --compute_lpips
```

### 随机种子说明
随机种子影响：
- PyTorch 的随机初始化（如果有）
- NumPy 的随机操作
- CUDA 的随机性

如果模型是完全确定性的（无dropout、无随机增强），不同种子的结果应该完全一致。

### 磁盘空间
- 每次推理会保存100张PNG图片（~20-50MB）
- CSV/JSON 文件很小（<1MB）
- 如需节省空间，可在评估后删除图片文件，保留 CSV/JSON

## 快速检查命令

```bash
# 查看最新的评估结果
ls -lt outputs/  # 按时间排序

# 快速查看指标汇总
cat outputs/zero_shot_*/metrics_summary.json | grep -A5 "psnr"

# 统计所有评估的平均PSNR
grep "Mean:" outputs/*/metrics_summary.json

# 查看最差的10张图片
head -11 outputs/*/metrics_per_image.csv | grep -v "image_name" | sort -t, -k2 -n
```

## 故障排除

### 问题：LPIPS 下载失败
**解决**：不使用 LPIPS 或使用代理/手动下载

### 问题：内存不足
**解决**：减少 batch_size（配置文件中）或使用更小的测试集

### 问题：outputs 目录被覆盖
**原因**：使用了旧版本的 infer.py
**解决**：确保使用最新版本（带时间戳的输出目录）

### 问题：多种子评估找不到输出目录
**原因**：eval_multi_seed.py 需要按照目录命名规则查找
**解决**：确保使用 infer.py 生成的标准输出目录（包含 _seed{N}）
