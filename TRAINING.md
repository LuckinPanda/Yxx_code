# 训练指南

## 快速开始

### 一键训练两个阶段

```bash
# 简单方式：自动按顺序训练 Stage-L → Stage-R-pre
python train_all.py
```

### 分开训练

```bash
# 只训练 Stage-L (照明网络预训练)
python train_all.py --stage L

# 只训练 Stage-R-pre (反射率预训练，需要illum_ckpt.pth)
python train_all.py --stage R-pre

# Stage-R-adapt 暂时舍弃（deprecated）
```

### 训练指定的组合

```bash
# 训练 Stage-L 和 Stage-R-pre
python train_all.py --stage L,R-pre

# Stage-R-adapt 暂时舍弃（deprecated）
```

## 详细说明

### Stage-L: 照明网络预训练

- **输入**: 配对的低光和正常光图像 (LSRW数据集)
- **训练**: 10个epoch，学习照明估计
- **输出**: `checkpoints/illum_ckpt.pth`
- **配置**: `configs/stage_L.yaml`

```bash
python train_all.py --stage L
```

### Stage-R-pre: 反射率预训练 (源域)

- **输入**: 源域低光图像 (LOL低光)
- **依赖**: 需要 `illum_ckpt.pth`
- **训练**: 30个epoch，Noise2Self风格自监督学习
- **损失函数**:
  - `L_ss`: mask像素重建损失（Noise2Self 核心损失）
  - `L_tv`: 输出反射率 R_e 的 TV 平滑先验（隐式去噪）
  - `L_cb`: 通道平衡先验 / 灰度世界假设（防止偏色）
  - `L_reg`: 温和的 delta 正则化
- **输出**: `checkpoints/denoise_pre_ckpt.pth`
- **配置**: `configs/stage_R_pre.yaml`

```bash
python train_all.py --stage R-pre
```

### Stage-R-adapt: 域适应（Deprecated）

- 暂时舍弃该阶段；相关脚本与配置保留，但不再用于当前实验流程。

## 自定义配置

如果需要使用不同的config文件路径：

```bash
# 自定义所有config路径
python train_all.py \
    --config-L configs/stage_L.yaml \
  --config-R-pre configs/stage_R_pre.yaml
```

## 训练检查点依赖关系

```
Stage-L (illum_ckpt.pth)
    ↓
Stage-R-pre (需要illum_ckpt.pth → denoise_pre_ckpt.pth)
```

### 依赖检查

脚本会自动检查所需的检查点：
- 如果选择训练R-pre但illum_ckpt.pth不存在，会报错
- Stage-R-adapt 暂时舍弃（deprecated）

## 推理

训练完成后，运行推理：

```bash
# Zero-shot mode (使用预训练的反射率网络)
# 默认开启灰度世界颜色校正
python infer.py --mode zero_shot --config configs/infer.yaml --seed 42

# 关闭颜色校正
python infer.py --mode zero_shot --config configs/infer.yaml --seed 42 --no_color_correct

# 调整颜色校正强度 (0=关闭, 1=全开, 默认0.8)
python infer.py --mode zero_shot --config configs/infer.yaml --seed 42 --color_strength 0.6

# Stage-R-adapt 暂时舍弃（deprecated）
```

## 推荐的工作流

### 首次训练

```bash
# 一键训练两个阶段
python train_all.py

# 等待训练完成...

# 运行推理评估（默认开启颜色校正）
python infer.py --mode zero_shot --config configs/infer.yaml --seed 42
```

### 快速迭代

```bash
# Stage-R-adapt 暂时舍弃（deprecated）
```

### 从头开始

```bash
# 完整重训
python train_all.py

# 或者逐个重训
python train_all.py --stage L
python train_all.py --stage R-pre  
# Stage-R-adapt 暂时舍弃（deprecated）
```

## 监控训练

训练过程中会输出：
- 每个epoch的平均损失
- 进度条显示训练进度
- 检查点保存位置
- 训练完成提示

所有日志也保存在 `logs/` 目录中：
- `logs/stage_L_*.log`
- `logs/stage_R_pre_*.log`
- Stage-R-adapt 暂时舍弃（deprecated）

## 常见问题

### Q: 如何只重训某一个阶段？
A: 使用 `--stage` 参数指定，例如 `python train_all.py --stage R-pre`

### Q: 如何跳过某个已训练好的阶段？
A: 只训练需要的阶段，例如 `python train_all.py --stage R-pre,R-adapt`

### Q: 检查点存储在哪里？
A: 默认在 `checkpoints/` 目录，可以在config文件中修改

### Q: 训练失败了怎么办？
A: 检查 `logs/` 目录中的日志文件，查看错误信息

## 参数说明

```
使用方法:
  python train_all.py [OPTIONS]

选项:
  --stage STAGE              要训练的阶段
                            - 'all' (默认): 训练两个阶段
                            - 'L': 只训练Stage-L
                            - 'R-pre': 只训练Stage-R-pre
                            - 'L,R-pre': 训练Stage-L和Stage-R-pre

  --config-L PATH           Stage-L的config文件 (默认: configs/stage_L.yaml)
  --config-R-pre PATH       Stage-R-pre的config文件 (默认: configs/stage_R_pre.yaml)
  
  -h, --help               显示帮助信息
```
