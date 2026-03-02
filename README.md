
# Retinex + Rotation-Equivariant AdaReNet

This project strictly follows `SPEC.md` (Final SPEC v4).

It implements a Retinex pipeline with a rotation-equivariant AdaReNet as the **only reflectance denoiser**, trained using **Noise2Self-style mask-based self-supervision** in the reflectance proxy domain.

Key features:
- **Guided-filter pre-denoising** of input before reflectance proxy computation
- **Noise2Self masked-pixel-only loss** for self-supervised denoising
- **Gradient preservation + color consistency + delta regularisation** losses
- **Flip augmentation** during training (rotation redundant with RotEqBlock)
- **Test-Time Augmentation (TTA)** via 4-fold flip ensemble
- **Post-processing pipeline**: bilateral filter, auto-contrast, unsharp mask
- **Gray-world color correction** post-processing to eliminate color cast
- **4 evaluation metrics**: PSNR, SSIM, MAE, LPIPS (all computed by default)

---

## Environment

```bash
conda activate adarenet
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `opencv-python`, `scipy`, `lpips`, `pyyaml`, `tqdm`

---

## Data Preparation

### Folder Layout

- Source paired (Stage-L):
  - `source_low_dir/`
  - `source_high_dir/`
  - Paired by identical filename stem.

- Source low-only (Stage-R-pre):
  - `source_low_dir/`

- Target low-only (inference):
  - `test_low_dir/`

Update dataset paths in `configs/*.yaml`.

---

## Training

### Stage-L (Illumination Pretraining)

```bash
python train_stage_L.py --config configs/stage_L.yaml
```

- Trains `IlluminationNet` only.
- AdaReNet is frozen.
- Saves `checkpoints/illum_ckpt.pth`.

---

### Stage-R-pre (Reflectance Pretraining, Source Low-Only)

```bash
python train_stage_R_pre.py --config configs/stage_R_pre.yaml
```

- Trains AdaReNet with **Noise2Self-style self-supervised loss**.
- Illumination branch is frozen.
- Reflectance proxy `P_ref` is constructed with guided-filter pre-denoising.
- **40 epochs**, lr=1e-4 (Adam, constant).
- **Flip augmentation**: random horizontal / vertical flip (configurable).
- **Gradient clipping**: max_norm=1.0 for training stability.
- **AMP** (mixed precision) on CUDA for speedup.
- **Pre-caching**: frozen illumination outputs computed once before training.
- Saves `checkpoints/denoise_pre_ckpt.pth`.

Self-supervision mechanism:

- Mixed training: 50% masked input (Noise2Self), 50% full input (bridges train-test gap)
- Sample binary mask `M ~ Bernoulli(0.2)`
- Construct masked reflectance:
  ```
  P_tilde = (1 - M) * P_ref + M * (local_mean + noise)
  ```
- Predict residual Δ
- Losses:
  ```
  L_ss    = masked-pixel-only L1 (Noise2Self guarantee)
  L_grad  = gradient-domain L1 (edge / structure preservation)
  L_color = channel-ratio color consistency (prevents color shift)
  L_reg   = mild delta regularization
  ```

---

### Stage-R-adapt (Target Low-Only Adaptation) — Deprecated

- 暂时舍弃该阶段；当前项目只使用 zero-shot 推理。
- 相关脚本与配置保留，但不再用于当前实验。

---

## Inference

```bash
# Basic inference with color correction + LPIPS + MAE (all default)
python infer.py --config configs/infer.yaml --mode zero_shot --seed 42

# Recommended: TTA + bilateral filter for best quality
python infer.py --mode zero_shot --seed 42 --tta --enhance --no_contrast --no_sharpen --bilateral_sc 100

# Full post-processing pipeline (bilateral + contrast + sharpen)
python infer.py --mode zero_shot --seed 42 --tta --enhance

# Disable color correction
python infer.py --config configs/infer.yaml --mode zero_shot --seed 42 --no_color_correct

# Adjust color correction strength (0=off, 1=full, default 0.3)
python infer.py --config configs/infer.yaml --mode zero_shot --seed 42 --color_strength 0.6

# Disable LPIPS (faster, no pretrained model download needed)
python infer.py --mode zero_shot --seed 42 --no_lpips
```

Forward pipeline:

```
I -> GuidedFilter(I, L_T) -> P_ref -> [TTA] -> Δ -> R_e -> I_hat
  -> GrayWorldCorrection -> [Bilateral -> Contrast -> Sharpen] -> Output
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tta` | off | 4-fold flip ensemble (TTA) |
| `--enhance` | off | Enable post-processing pipeline |
| `--no_bilateral` | (on when enhance) | Disable bilateral denoising |
| `--no_contrast` | (on when enhance) | Disable contrast stretching |
| `--no_sharpen` | (on when enhance) | Disable unsharp mask |
| `--bilateral_sc` | 100 | Bilateral filter sigmaColor |
| `--smooth_illum` | 0 | Gaussian sigma for L_T smoothing |
| `--no_color_correct` | off | Disable gray-world color correction |
| `--color_strength` | 0.3 | Color correction blend strength |
| `--no_lpips` | off | Disable LPIPS metric |
| `--seed` | None | Random seed for reproducibility |

---

## Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| omega | 15.0 | Gamma correction exponent |
| tau | 0.1 | Illumination lower bound |
| eps | 1e-6 | Numerical stability |
| pref_max | 3.0 / 5.0 | P_ref clamp upper bound (infer/train) |
| guided_filter_radius | 3 | Pre-denoising filter radius |
| guided_filter_eps | 0.02 | Pre-denoising regularization |
| mask_prob | 0.2 | Masking probability |
| full_input_ratio | 0.5 | Fraction of full-input batches |
| color_strength | 0.3 | Gray-world correction strength |
| bilateral_sc | 100 | Bilateral filter sigmaColor |

**All constants must be consistent across training and inference.**

---

## Notes (SPEC Constraints)

- Float32 in [0,1]
- Output clamped to [0,1]
- `L_T`/`L_e`: 1-channel
- `P_ref`: 3-channel
- AdaReNet input: concat(P_ref, L_e) = 4 channels
- AdaReNet output: residual Δ
- Mixed-input training: 50% masked (Noise2Self) + 50% full (bridge train-test gap)
- Inference uses the zero-shot (pretrained) AdaReNet weights
- Metrics: PSNR, SSIM, MAE, LPIPS (all default; LPIPS disable with `--no_lpips`)
