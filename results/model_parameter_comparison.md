# Model Parameter Comparison: GETRegionPretrain vs GETRegionDiffusion

## Executive Summary

This document compares the parameter counts between two models used in the pretraining scripts:
- **GETRegionPretrain** (from `scripts/run_pretrain.py`)
- **GETRegionDiffusion** (from `scripts/run_diffusion-transformer.py`)

## Overall Parameter Counts

| Model | Total Parameters | Difference |
|-------|-----------------|------------|
| **GETRegionPretrain** | **85.48M** (85,484,059) | Baseline |
| **GETRegionDiffusion** | **128.77M** (128,767,003) | +43.28M (+50.63%) |

The diffusion model has **50.63% more parameters** than the standard transformer model.

## Component Breakdown

### Shared Components (Same in Both Models)

| Component | Parameters | Description |
|-----------|------------|-------------|
| `region_embed` | 218,112 | Linear projection: 283 → 768 |
| `head_mask` | 217,627 | Linear projection: 768 → 283 |
| `mask_token` | 768 | Learnable mask token embedding |
| `cls_token` | 768 | CLS token embedding |

**Total shared:** 437,275 parameters

### Encoder Comparison

The main difference lies in the encoder architecture:

#### GETTransformer (GETRegionPretrain)
- **Total:** 85.05M parameters
- **Architecture:** Standard Transformer with 12 blocks
- **Per Block:** ~7.09M parameters
  - `norm1`: 1,536 params (LayerNorm with affine)
  - `attn`: 2.36M params
    - `qkv`: 1.77M params (768 → 2304)
    - `proj`: 590,592 params (768 → 768)
  - `norm2`: 1,536 params (LayerNorm with affine)
  - `mlp`: 4.72M params
    - `fc1`: 2.36M params (768 → 3072)
    - `fc2`: 2.36M params (3072 → 768)

#### GETRegionDiTEncoder (GETRegionDiffusion)
- **Total:** 128.33M parameters
- **Architecture:** Diffusion Transformer (DiT) with timestep conditioning
- **Timestep Embedder:** 787,968 parameters
  - Sinusoidal embedding (256 dims)
  - MLP: Linear(256 → 768) + SiLU + Linear(768 → 768)
- **Per DiT Block:** ~10.63M parameters
  - `norm1`: 0 params (LayerNorm **without** affine)
  - `attn`: 2.36M params (same as standard transformer)
  - `norm2`: 0 params (LayerNorm **without** affine)
  - `mlp`: 4.72M params (same as standard transformer)
  - `adaLN_modulation`: **3.54M params** (NEW)
    - SiLU + Linear(768 → 4608) producing 6 modulation vectors

**Encoder Difference:** +43.28M parameters (+50.89%)

## Key Architectural Differences

### 1. Timestep Conditioning
The diffusion model adds a **TimestepEmbedder** that converts scalar timesteps into conditioning vectors:
- Input: Integer timestep `t ∈ [0, 1000]`
- Output: Conditioning vector `c ∈ ℝ^768`
- Parameters: 787,968

### 2. Adaptive Layer Normalization (adaLN)
Each DiT block uses **adaptive layer normalization** instead of standard LayerNorm:
- **Standard Transformer:** Uses LayerNorm with learnable affine parameters (γ, β)
- **DiT Block:** Uses LayerNorm **without** affine parameters, but predicts 6 modulation vectors dynamically:
  - `shift_msa`, `scale_msa`, `gate_msa` (for attention)
  - `shift_mlp`, `scale_mlp`, `gate_mlp` (for MLP)
- Each modulation vector has dimension 768
- Total per block: 6 × 768 = 4,608 parameters predicted from conditioning
- Linear layer: 768 → 4,608 = **3,543,552 parameters per block**
- Across 12 blocks: **42.52M additional parameters**

### 3. Parameter Savings
- DiT blocks save parameters by removing affine parameters from LayerNorm:
  - Per block: 2 × 1,536 = 3,072 parameters saved
  - Across 12 blocks: 36,864 parameters saved
- However, this is far outweighed by the adaLN_modulation parameters

## Parameter Distribution

```
GETRegionPretrain:
├── region_embed:     0.26% (218K)
├── encoder:         99.49% (85.05M)
├── head_mask:        0.25% (218K)
└── tokens:           0.00% (1.5K)

GETRegionDiffusion:
├── region_embed:     0.17% (218K)
├── encoder:         99.66% (128.33M)
│   ├── t_embedder:   0.61% (788K)
│   └── DiT blocks:  99.05% (127.54M)
├── head_mask:        0.17% (218K)
└── tokens:           0.00% (1.5K)
```

## Why the Difference?

The diffusion model needs additional parameters to:
1. **Condition on timesteps:** The timestep embedder allows the model to learn different behaviors at different noise levels
2. **Adaptive normalization:** The adaLN mechanism enables dynamic modulation of layer activations based on the diffusion timestep, which is crucial for learning the denoising process

This architectural choice follows the DiT (Diffusion Transformer) design pattern, which has been shown to be effective for diffusion-based generative modeling.

## Implications

1. **Memory:** The diffusion model requires ~50% more GPU memory for training
2. **Training Time:** Forward/backward passes will be slower due to increased parameters
3. **Expressiveness:** The additional parameters enable the model to learn timestep-dependent transformations, which is essential for diffusion-based generation

