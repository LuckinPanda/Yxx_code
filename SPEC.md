
# Cross-Domain Low-Light Enhancement
## Retinex Framework with Rotation-Equivariant AdaReNet Denoiser
### (Final SPEC v3 – Goal-Aligned & Unambiguous)

---

## 0. Design Goal (Frozen)

The **sole objective** of this project is:

> **To integrate a rotation-equivariant denoising network (AdaReNet) into a Retinex-based framework, such that it serves as a stable reflectance denoiser / restorer under cross-domain low-light conditions.**

All architectural, training, and loss-function decisions in this document are made **only** to support this goal.

---

## 1. Problem Definition

Given a low-light image \(I\), the goal is to produce an enhanced image \(\hat{I}\) that:

- Has corrected illumination and reduced noise,
- Remains stable under domain shifts (camera / sensor / noise statistics),
- Does **not** rely on normal-light ground truth in the target domain.

---

## 2. Image Formation Model (Retinex)

Low-light image formation follows the Retinex model:

\[
I = R \odot L + n
\]

where:
- \(R\): reflectance (structure, texture, color),
- \(L\): illumination (intensity field),
- \(n\): noise / residual.

Noise is not explicitly estimated, but absorbed into a **residual prediction** performed in the reflectance domain.

---

## 3. Illumination Prior via Retinex Unfolding

### 3.1 Illumination Estimation

Illumination is estimated by a dilated-convolution network with residual connections:

\[
L_T = \text{IlluminationNet}(I)
\]

- Shape: \(L_T \in \mathbb{R}^{B \times 1 \times H \times W}\)
- Architecture: dilated conv layers (dilation 1→2→4→8→1) + skip connections
- Activation: LeakyReLU (hidden), Sigmoid (output) → \(L_T \in [0,1]\)
- Learned on paired source-domain data
- **Frozen** in all stages except illumination pretraining

### 3.2 Illumination Adjustment

A **gamma-correction** illumination amplification is applied:

\[
L_e = \text{clamp}\!\left(L_T^{\,1/\omega},\; 0,\; 1\right)
\]

- \(\omega\) is a fixed hyperparameter (default: 15.0)
- Gamma correction provides non-linear dynamic-range expansion:
  dark regions are lifted substantially while bright regions remain stable
- \(L_e\) is not learnable and does not change across stages
- **The same \(\omega\) must be used in training and inference**

---

## 4. Reflectance Proxy Construction

To decouple noise from illumination, a reflectance proxy is constructed.

### 4.1 Guided Filter Pre-Denoising

Before computing the reflectance proxy, the low-light input is pre-filtered
using a **guided filter** with \(L_T\) as the guidance image:

\[
I_{filtered} = \text{GuidedFilter}(I,\; L_T,\; r,\; \epsilon_{gf})
\]

- \(r = 3\) (filter radius), \(\epsilon_{gf} = 0.02\) (regularization)
- This reduces noise **before** the division amplifies it,
  while preserving edges aligned with illumination structure.
- Implemented as box-window guided filter (GPU-friendly, no external deps).

### 4.2 Illumination Lower Bound

\[
\tilde{L}_T = \max(L_T, \tau), \quad \tau = 0.1
\]

### 4.3 Reflectance Proxy

\[
P_{ref} = \text{clamp}\!\left(\frac{I_{filtered}}{\tilde{L}_T + \varepsilon},\; 0,\; P_{max}\right), \quad \varepsilon = 10^{-6},\; P_{max} = 3.0
\]

- \(P_{ref} \in \mathbb{R}^{B \times 3 \times H \times W}\)
- Broadcast division is used
- \(P_{max} = 3.0\) prevents numerical explosion in dark regions
  (lowered from 5.0 to further suppress noise amplification)

---

## 5. Reflectance Denoising with Rotation-Equivariant AdaReNet

### 5.1 Role of AdaReNet

AdaReNet is used **exclusively** as a reflectance denoiser / restorer inside the Retinex framework.

> AdaReNet follows the original **rotation-equivariant adaptive denoising design**, and is treated as a fixed-structure denoiser in this project.

Its purpose is to provide a **strong architectural prior** under self-supervised and cross-domain conditions.

---

### 5.2 Internal Structure Clarification (Critical)

AdaReNet internally consists of:

- A **Vanilla convolutional branch**,
- A **Rotation-Equivariant (EQ) branch**,
- An internal **Fusion MaskNetwork** that adaptively combines the two branches.

**Important clarification**:

- This *Fusion MaskNetwork* is a **structural gating mechanism**,  
- It is **part of the AdaReNet architecture**,  
- It is **NOT** a blind-spot or masked-pixel self-supervision mechanism.

The Fusion MaskNetwork **is allowed and required** when using AdaReNet.

---

### 5.3 Network Input and Output

**Input to AdaReNet**:
```text
Input = concat(P_ref, L_e)   # 4 channels
```

**Output**:
```text
Output = Δ                  # residual (noise / unwanted components)
```

Residual prediction is mandatory.

---

### 5.4 Reflectance Recovery and Reconstruction

\[
R_e = P_{ref} - \Delta
\]

\[
\hat{I} = \text{clip}(R_e \odot L_e)
\]

- Entire project operates in \([0, 1]\) float space
- Final output is clipped to \([0, 1]\)

---

## 6. Self-Supervised Denoising Objective

### 6.1 Principle

Self-supervised denoising in this framework follows a **mask-based inpainting supervision strategy**, 
which prevents identity mapping while preserving structural consistency.

The supervision signal is constructed directly in the reflectance proxy domain.

---

### 6.2 Masked Reflectance Construction

For each reflectance proxy \(P_{ref}\), a binary mask \(M\) is sampled:

\[
M(i,j) \sim \text{Bernoulli}(p)
\]

Masked reflectance is constructed as:

\[
\tilde{P}_{ref}
=
(1-M) \odot P_{ref}
+
M \odot \xi
\]

where:

- \(M \in \{0,1\}^{H \times W}\),
- \(p\) is the masking probability (e.g., 0.1–0.3),
- \(\xi\) is random noise or random neighbor sampling,
- Masking is spatial and independent per batch.

---

### 6.3 Residual Prediction and Noise2Self-style Masked Loss

AdaReNet predicts residual **always on masked input** (no mixed full-input training):

\[
\Delta = G_R(\tilde{P}_{ref}, L_e)
\]

Recovered reflectance:

\[
\tilde{R}_e = \tilde{P}_{ref} - \Delta
\]

Self-supervised loss is computed **only on masked pixels** (Noise2Self principle):

\[
\mathcal{L}_{ss}
=
\frac{\sum M \odot |\tilde{R}_e - P_{ref}|}{\sum M + 10^{-6}}
\]

This forces the network to predict missing reflectance from **neighbors only**,
and by the Noise2Self theorem the optimal prediction converges to the
clean signal rather than the noisy input.

**Key design choices (v2)**:
- 100% masked input (no 50% full-input mode which caused identity-mapping / noise retention)
- Mask probability \(p = 0.3\) (increased from 0.2 for stronger denoising signal)

---

### 6.4 Auxiliary Losses

In addition to the primary masked loss, two self-supervised priors are used:

**TV smoothness on recovered reflectance:**

\[
\mathcal{L}_{tv} = \|\nabla_h \tilde{R}_e\|_1 + \|\nabla_w \tilde{R}_e\|_1
\]

Encourages spatial smoothness of the output as an implicit denoising prior.
Unlike the previous `gradient_loss(pred, noisy_P_ref)`, this does **not** force
the output to replicate noisy gradients.

**Channel balance prior (gray-world):**

\[
\mathcal{L}_{cb} = \|\mu_{ch}(\tilde{R}_e) - \bar{\mu}(\tilde{R}_e)\|_1
\]

Prevents systematic color shift (e.g. green tint from single-channel \(L_T\) division)
without requiring a clean reference image.

---

### 6.5 Training Mechanism Clarification

- Masking is **always** applied to the input reflectance proxy.
- Loss is computed **only** on masked pixels.
- No paired clean image is used for denoising supervision.
- Illumination branch remains frozen.
- AdaReNet structure remains unchanged.
- **No** `gradient_loss(pred, target)` or `color_consistency_loss(pred, target)`
  against noisy \(P_{ref}\) (these were removed as they forced noise retention).

The masking mechanism is used strictly as a self-supervised signal,
and does not alter the inference architecture.


## 7. Training Protocol

### Stage-L: Illumination Pretraining (Source Domain)

- Data: paired low / normal images
- Trainable: illumination network
- Frozen: AdaReNet
- Pseudo-GT: \(L_{pseudo} = \text{avg\_pool}\!\left(\text{mean}_c\!\left(\frac{I_{low}}{I_{high} + \varepsilon}\right)\right)\)
- Loss: \(\mathcal{L}_{illum} = \lambda_1 \|L_T - L_{pseudo}\|_1 + \lambda_{tv} \text{TV}(L_T) + \lambda_{rec} \|L_T \odot I_{high} - I_{low}\|_1\)
- Output: `illum_ckpt.pth`

---

### Stage-R-pretrain: Reflectance Pretraining (Source Domain)

- Data: source-domain low-only images
- Frozen: illumination network
- Trainable: AdaReNet
- Epochs: 30
- Loss: \(\mathcal{L}_{ss} + \lambda_{tv} \mathcal{L}_{tv} + \lambda_{cb} \mathcal{L}_{cb} + \lambda_{\delta} \|\Delta\|_1\)
  - \(\mathcal{L}_{ss}\): Noise2Self masked-pixel-only inpainting loss (§6.3)
  - \(\mathcal{L}_{tv}\): TV smoothness on recovered reflectance (\(\lambda_{tv} = 0.05\))
  - \(\mathcal{L}_{cb}\): channel balance prior / gray-world (\(\lambda_{cb} = 0.1\))
  - \(\|\Delta\|_1\): mild residual regularisation (\(\lambda_{\delta} = 0.05\))
- Output: `denoise_pre_ckpt.pth`

---

### Stage-R-adapt: Target-Domain Adaptation (Deprecated)

- 暂时舍弃该阶段；当前项目只采用 zero-shot 推理。
- 相关脚本与配置保留，但不作为当前实验流程的一部分。

---

## 8. Inference Mode (Zero-shot)

- Inference uses the pretrained AdaReNet weights \(\theta_F^{pre}\) (zero-shot only).
- The forward pipeline is identical to training-time construction.
- **Post-processing**: Gray-world color correction is applied to the output:
  \[
  \hat{I}_{corrected}(c) = \hat{I}(c) \cdot \frac{\bar{\mu}}{\mu_c + \epsilon}
  \]
  where \(\mu_c\) is the per-channel mean and \(\bar{\mu}\) is the global mean.
  - Strength parameter (default 0.8) blends between original and corrected output.
  - Can be disabled with `--no_color_correct` CLI flag.

---

## 9. Ablation Design (Goal-Oriented)

- Remove rotation-equivariant branch (Vanilla only)
- EQ-only branch (no Vanilla)
- Remove Fusion MaskNetwork (fixed fusion)
- Unfreeze illumination during training
- Adapt/anchor 相关消融目前暂停

Each ablation answers **whether rotation equivariance improves stability when embedded in Retinex**.

---

## 10. Final Constraints

- AdaReNet is the **only** reflectance denoiser
- Rotation equivariance is a **core prior**
- Mask-based Noise2Self self-supervision for denoising (no paired clean images)
- No target-domain GT usage
- No architectural difference between training and inference
- **\(\omega\) must be identical across all stages and inference**
- **Guided filter parameters must be consistent across training and inference**

---

**This document is the single source of truth for implementation and experimentation.**
