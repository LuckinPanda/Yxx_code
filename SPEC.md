
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

### 4.1 Illumination Lower Bound

\[
\tilde{L}_T = \max(L_T, \tau), \quad \tau = 10^{-3}
\]

### 4.2 Reflectance Proxy

\[
P_{ref} = \frac{I}{\tilde{L}_T + \varepsilon}, \quad \varepsilon = 10^{-6}
\]

- \(P_{ref} \in \mathbb{R}^{B \times 3 \times H \times W}\)
- Broadcast division is used
- No clipping is applied at this stage

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

### 6.3 Residual Prediction and Masked Loss

AdaReNet predicts residual:

\[
\Delta = G_R(\tilde{P}_{ref}, L_e)
\]

Recovered reflectance:

\[
\tilde{R}_e = \tilde{P}_{ref} - \Delta
\]

Self-supervised loss is computed **only on masked pixels**:

\[
\mathcal{L}_{ss}
=
\left\|
M \odot (\tilde{R}_e - P_{ref})
\right\|_1
\]

This forces the network to predict missing reflectance structure
rather than copying the input.

---

### 6.4 Training Mechanism Clarification

- Masking is applied only to the input reflectance proxy.
- Loss is computed only on masked pixels.
- No paired clean image is used.
- Illumination branch remains frozen.
- AdaReNet structure remains unchanged.

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
- Loss: \(\mathcal{L}_{ss} + \lambda_{grad} \mathcal{L}_{grad} + \lambda_{color} \mathcal{L}_{color} + \lambda_{\delta} \|\Delta\|_1\)
  - \(\mathcal{L}_{ss}\): masked inpainting loss (primary, §6.3)
  - \(\mathcal{L}_{grad}\): gradient-domain L1 for edge preservation
  - \(\mathcal{L}_{color}\): channel-ratio consistency for color stability
  - \(\|\Delta\|_1\): residual regularisation
- Output: `denoise_pre_ckpt.pth`

---

### Stage-R-adapt: Target-Domain Adaptation (Deprecated)

- 暂时舍弃该阶段；当前项目只采用 zero-shot 推理。
- 相关脚本与配置保留，但不作为当前实验流程的一部分。

---

## 8. Inference Mode (Zero-shot)

- Inference uses the pretrained AdaReNet weights \(\theta_F^{pre}\) (zero-shot only).
- The forward pipeline is identical to training-time construction.

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
- No blind-spot / masked-pixel self-supervision
- No target-domain GT usage
- No architectural difference between training and inference
- **\(\omega\) must be identical across all stages and inference**

---

**This document is the single source of truth for implementation and experimentation.**
