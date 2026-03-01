
# Retinex + Rotation-Equivariant AdaReNet

This project strictly follows `SADD/SPEC.md` (Final SPEC v3).

It implements a Retinex pipeline with a rotation-equivariant AdaReNet as the **only reflectance denoiser**, trained using **mask-based self-supervision** in the reflectance proxy domain.

The only architectural mask is AdaReNet's internal Fusion MaskNetwork (structural gating).  
Additionally, masked-pixel supervision is used as the self-supervised training signal.

---

## Environment

```bash
conda activate uretinex-gpu
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

- Trains AdaReNet with **mask-based self-supervised loss** `L_mask`.
- Illumination branch is frozen.
- Reflectance proxy `P_ref` is constructed as in SPEC.
- Saves `checkpoints/denoise_pre_ckpt.pth`.

Self-supervision mechanism:

- Sample binary mask `M`
- Construct masked reflectance:
  ```
  P_tilde = (1 - M) * P_ref + M * ξ
  ```
- Predict residual Δ
- Compute masked loss:
  ```
  L_mask = | M * (R_tilde - P_ref) |_1
  ```

---

### Stage-R-adapt (Target Low-Only Adaptation) — Deprecated

- 暂时舍弃该阶段；当前项目只使用 zero-shot 推理。
- 相关脚本与配置保留，但不再用于当前实验。

---

## Inference

```bash
python infer.py --config configs/infer.yaml --mode zero_shot
```

Forward pipeline:

```
I -> L_T -> L_e -> P_ref -> Δ -> R_e -> I_hat
```

---

## Notes (SPEC Constraints)

- Float32 in [0,1]
- Output clamped to [0,1]
- `L_T`/`L_e`: 1-channel
- `P_ref`: 3-channel
- Constants:
  - omega = 2.0
  - tau = 1e-3
  - eps = 1e-6
- AdaReNet input: concat(P_ref, L_e)
- AdaReNet output: residual Δ
- Mask probability configurable (e.g., 0.1–0.3)
- Loss computed only on masked pixels

---

Inference uses the zero-shot (pretrained) AdaReNet weights.
