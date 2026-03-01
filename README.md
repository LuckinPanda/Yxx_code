
# Retinex + Rotation-Equivariant AdaReNet

This project strictly follows `SPEC.md` (Final SPEC v4).

It implements a Retinex pipeline with a rotation-equivariant AdaReNet as the **only reflectance denoiser**, trained using **Noise2Self-style mask-based self-supervision** in the reflectance proxy domain.

Key features:
- **Guided-filter pre-denoising** of input before reflectance proxy computation
- **Noise2Self masked-pixel-only loss** for self-supervised denoising
- **TV smoothness + channel balance priors** (replacing noise-preserving losses)
- **Gray-world color correction** post-processing to eliminate color cast

---

## Environment

```bash
conda activate adarenet
pip install -r requirements.txt
```

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
- Saves `checkpoints/denoise_pre_ckpt.pth`.

Self-supervision mechanism (Noise2Self):

- Sample binary mask `M ~ Bernoulli(0.3)`
- Construct masked reflectance:
  ```
  P_tilde = (1 - M) * P_ref + M * (local_mean + noise)
  ```
- Predict residual Δ (always from masked input)
- Losses:
  ```
  L_ss   = masked-pixel-only L1 (Noise2Self guarantee: converges to clean signal)
  L_tv   = TV smoothness on R_e (implicit denoising prior)
  L_cb   = channel balance / gray-world prior (prevents color shift)
  L_reg  = mild delta regularization
  ```

---

### Stage-R-adapt (Target Low-Only Adaptation) — Deprecated

- 暂时舍弃该阶段；当前项目只使用 zero-shot 推理。
- 相关脚本与配置保留，但不再用于当前实验。

---

## Inference

```bash
# Basic inference with color correction (default)
python infer.py --config configs/infer.yaml --mode zero_shot --seed 42

# Disable color correction
python infer.py --config configs/infer.yaml --mode zero_shot --seed 42 --no_color_correct

# Adjust color correction strength (0=off, 1=full, default 0.8)
python infer.py --config configs/infer.yaml --mode zero_shot --seed 42 --color_strength 0.6
```

Forward pipeline:

```
I -> GuidedFilter(I, L_T) -> P_ref -> Δ -> R_e -> I_hat -> GrayWorldCorrection -> Output
```

---

## Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| omega | 15.0 | Gamma correction exponent |
| tau | 0.1 | Illumination lower bound |
| eps | 1e-6 | Numerical stability |
| pref_max | 3.0 | P_ref clamp upper bound |
| guided_filter_radius | 3 | Pre-denoising filter radius |
| guided_filter_eps | 0.02 | Pre-denoising regularization |
| mask_prob | 0.3 | Masking probability |

**All constants must be consistent across training and inference.**

---

## Notes (SPEC Constraints)

- Float32 in [0,1]
- Output clamped to [0,1]
- `L_T`/`L_e`: 1-channel
- `P_ref`: 3-channel
- AdaReNet input: concat(P_ref, L_e) = 4 channels
- AdaReNet output: residual Δ
- Loss computed only on masked pixels (Noise2Self)
- Inference uses the zero-shot (pretrained) AdaReNet weights
