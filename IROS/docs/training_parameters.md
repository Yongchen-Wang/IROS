# Training parameters

All values follow the IROS paper (Sec. III-E and IV-C). They are the
defaults of `training/core/train_model.py`; anything can be overridden on
the command line.

## Data

| Parameter | Value | Notes |
|---|---|---|
| `window_size` (f) | 4 | temporal sliding window of consecutive frames |
| `img_size` | 224 × 224 | ImageNet-standard input for the ResNet-50 backbone |
| `chunk_size` (C) | 5 | future steps predicted per forward pass; Table IV evaluates 1 / 5 / 10 |
| split | Task 1 & 3 → train; Task 2 → val/test | Task 2 divided temporally into halves; overall ≈ 0.7 : 0.15 : 0.15, preventing leakage between splits |

## Model (Bi-CAST)

| Parameter | Value |
|---|---|
| Visual stream | ResNet-50 (ImageNet) → 2048 → linear 1024 + sinusoidal PE; Table IV swaps ResNet-18/34/50 inside Bi-CAST |
| Intent stream | bilateral `[F, ΔF, σ]` (6) → MLP → 128 |
| Safety stream | `[IoU, d_min]` (2) → MLP → 64 |
| Transformer encoder | 8 layers, d_model 1216, 8 heads, GELU |
| Heads | two independent MLPs → C = 5 chunks per arm |
| Output bound | `0.9 · sigmoid` → α ∈ [0, 0.9] (≥ 10 % human authority, Eq. 7) |

## Loss (Eq. 9)

| Parameter | Value | Notes |
|---|---|---|
| main term | Huber, δ = 0.1 | per arm |
| `w_L` / `w_R` | 0.6 / 0.4 | Optuna hyper-parameter search; higher left-arm weight compensates for its more frequent occlusion |
| `lambda_s` | 0.1 | smoothness penalty on consecutive chunk predictions |
| `lambda_1` | 1e-3 | L1 regularisation over model parameters (Σ\|θ\|) |

## Optimisation

| Parameter | Value |
|---|---|
| optimiser | AdamW |
| learning rate | 1e-4, cosine annealing (T_max = epochs) |
| weight decay | 1e-4 |
| batch size | 32 |
| epochs | 60 |
| hardware (paper) | 2 × NVIDIA A100, 80 GB each |

## Tracker (YOLOv8s-seg, `training/scripts/train_yolo_seg.py`)

| Parameter | Value |
|---|---|
| starting weights | COCO-pretrained YOLOv8s-seg |
| epochs / batch / input | 30 / 64 / 640 × 640 |
| optimiser | SGD, momentum 0.937 |
| learning rate | 0.01 → 0.001 via cosine annealing (`lr0=0.01`, `lrf=0.1`, `cos_lr`) |
| augmentation | HSV jittering, random scaling, mosaic (ultralytics defaults) |
| dataset | 1,500 images, split 8 : 1 : 1 |

## Paper Table IV switches

`training/core/train_model.py` exposes the paper ablations directly:

| Ablation | CLI | Paper settings |
|---|---|---|
| visual backbone | `--bicast_backbone` | `resnet18`, `resnet34`, `resnet50` |
| input modality | `--input_modality` | `vision`, `vision_fsr`, `vision_fsr_safety` |
| smoothness | `--lambda_s` | `0` vs default `0.1` |
| chunk size | `--chunk_size` | `1`, `5`, `10` |
