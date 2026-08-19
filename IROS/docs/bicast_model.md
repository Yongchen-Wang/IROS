# Bi-CAST model (`src/iros/models/bicast.py`)

Reference implementation of the authority-negotiation network from
*"Context-Aware Adaptive Shared Control for Magnetically-Driven Bimanual
Dexterous Micromanipulation"* (IROS 2026). Section numbers below refer to the
paper.

## Inputs (temporal window f = 4 frames)

| Stream | Per-frame input | Encoder | Per-frame dim |
|---|---|---|---|
| Visual | RGB frame 224×224×3 | ResNet-50 (ImageNet) → 2048 → linear projection | 1024 (+ sinusoidal positional embeddings) |
| User intent | bilateral FSR features `[F_t, ΔF_t, σ_t]` × 2 arms | MLP (6 → 64 → 128) | 128 |
| Safety | `[IoU_t, d_t^min]` | MLP (2 → 32 → 64) | 64 |

The visual and intent features are concatenated per frame and fed together
with the safety features into an **eight-layer Transformer encoder**
(d_model = 1024 + 128 + 64 = 1216, 8 heads, GELU). The encoded sequence is
temporally mean-pooled, and **two independent MLP heads** decode left/right
authority chunks of **C = 5** future steps, each bounded by a scaled sigmoid
(Eq. 7):

```
α̂_{t,i,k} = 0.9 · σ(logit_{t,i,k}),   i ∈ {L, R}
```

so the operator always retains at least 10 % control authority.

The intent signal I(t) entering the FSR features is the per-session
calibrated, w = 3 sliding-window-averaged force (Eq. 4–5); see
`tools/data/import_wall_distance_fsr.py`. `d_t^min` comes from the phantom
free-space distance transform (`tools/data/calculate_wall_distance.py`),
and `IoU_t` from the segmentation tracker.

## Causal chunk aggregation (Eq. 8)

At run time the executed authority ensembles the overlapping chunk
predictions that target the current step, produced by the C most recent
forward passes:

```
α_t^i = Σ_{k=0}^{C−1} ω_k · α̂_{t−k, i, k}
```

`ChunkAggregator` implements this with exponential weights
`ω_k ∝ exp(−k/2)` (normalised), which emphasise the most recent forward
pass (k = 0) while smoothing over the older overlapping predictions, so
that momentary noise in any single pass cannot cause a sudden jump in the
executed authority. See `examples/bicast_inference.py` for control-loop
usage together with the blending law (Eq. 6).

## Training objective (Eq. 9)

```
L = Σ_{i∈{L,R}} w_i · [ L_Huber^i + λ_s Σ_{k=0}^{C−2} (α̂_{k+1}^i − α̂_k^i)² ]
    + λ_1 Σ |θ|
```

with `w_L = 0.6`, `w_R = 0.4` (Optuna hyper-parameter search; the higher
left-arm weight compensates for its more frequent occlusion), Huber
δ = 0.1, `λ_s = 0.1`, and `λ_1 = 1e-3` (literal sum of absolute parameter
values). Implemented in `BiCASTLoss`.

## Training setup (Sec. IV-C)

* AdamW, lr = 1e-4, weight decay = 1e-4, cosine annealing
* batch size 32, 60 epochs
* data split: Task 1 & Task 3 recordings → training; Task 2 recording →
  validation/test (temporal halves), overall ≈ 0.7 : 0.15 : 0.15
* hardware used in the paper: 2 × NVIDIA A100 (80 GB)

Baselines for the network comparison (Table III) live in
`training/core/train_model.py` under the same output contract:
`three_layer_cnn`, `resnet18`, `resnet50`, `tsm_resnet18`, `r2plus1d`,
`mvit`, `vit`.  Table-III evaluation reports the left/right smoothness
columns independently.

## Table-IV ablations

The `BiCAST` constructor exposes only the factors varied in Table IV while
keeping the rest of the model fixed: `backbone_name` (`resnet18`, `resnet34`,
`resnet50`), `use_fsr`, `use_safety`, and `chunk_size` (`1`, `5`, `10`).
The training CLI maps the modality rows exactly as `vision`, `vision_fsr`,
and `vision_fsr_safety`.  See
`training/scripts/run_table_iv_ablations.py` for the paper-ordered sweep.
