#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torchvision.models as tv_models
import torchvision.models.video as tv_video
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from iros.models.bicast import (              
    BiCAST,
    BiCASTLoss,
    ChunkAggregator,
    compute_fsr_features,
)

WINDOW = 4                           
CHUNK = 5                       
IMG_SIZE = 224
ALPHA_MAX = 0.9                                             

REQUIRED_LABEL_KEYS = ("alpha_L", "alpha_R", "fsr_L", "fsr_R", "iou", "d_wall")


def _finite_float32_label(values, key: str) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"alpha_labels.pkl[{key!r}] must be a one-dimensional numeric array"
        ) from exc

    if array.ndim != 1:
        raise ValueError(
            f"alpha_labels.pkl[{key!r}] must be one-dimensional, got shape {array.shape}"
        )
    invalid = ~np.isfinite(array)
    if invalid.any():
        positions = np.flatnonzero(invalid)
        examples = ", ".join(str(int(pos)) for pos in positions[:5])
        raise ValueError(
            f"alpha_labels.pkl[{key!r}] contains {len(positions)} non-finite "
            f"value(s) at indices {examples}"
        )
    return array

class AuthorityDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        frame_range: Optional[tuple[int, int]] = None,
        img_size: int = IMG_SIZE,
        fsr_var_window: int = 5,
        chunk_size: int = CHUNK,
    ):
        self.root = Path(root)
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        img_dir = self.root / "images"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"images/ not found under {self.root}")
        self.img_files = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png")
        )

        label_file = self.root / "alpha_labels.pkl"
        if not label_file.exists():
            raise FileNotFoundError(f"{label_file} not found")
        with open(label_file, "rb") as f:
            lab = pickle.load(f)

        if not isinstance(lab, dict):
            raise ValueError(f"{label_file} must contain a dictionary")
        missing = [key for key in REQUIRED_LABEL_KEYS if key not in lab]
        if missing:
            raise ValueError(
                f"{label_file} is missing required arrays: {', '.join(missing)}"
            )
        arrays = {
            key: _finite_float32_label(lab[key], key) for key in REQUIRED_LABEL_KEYS
        }

        n = min(
            len(self.img_files),
            *(len(arrays[key]) for key in REQUIRED_LABEL_KEYS),
        )
        if n == 0:
            raise ValueError(f"{self.root} contains no aligned image/label samples")
        self.alpha = np.stack(
            [arrays["alpha_L"][:n], arrays["alpha_R"][:n]], axis=1
        )

        fsr_raw = np.stack(
            [arrays["fsr_L"][:n], arrays["fsr_R"][:n]], axis=1
        )
        self.fsr_feats = compute_fsr_features(fsr_raw, var_window=fsr_var_window)
        if not np.isfinite(self.fsr_feats).all():
            raise ValueError(
                "computed FSR features contain non-finite values; "
                "check the magnitude of fsr_L/fsr_R"
            )

        iou = arrays["iou"][:n]
        d_wall = arrays["d_wall"][:n]
        d_wall_scale = max(float(np.max(d_wall)), 1e-6)
        d_norm = d_wall / d_wall_scale
        if not np.isfinite(d_norm).all():
            raise ValueError("normalized d_wall contains non-finite values")
        self.safety = np.stack([iou, d_norm], axis=1)

        lo, hi = frame_range if frame_range else (0, n)
        lo, hi = max(lo, 0), min(hi, n)

        self.frame_range = (lo, hi)
        self.indices = list(
            range(lo + WINDOW - 1, hi - self.chunk_size + 1)
        )
        self.n = n

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((img_size, img_size), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _load_frame(self, idx: int) -> torch.Tensor:
        img = cv2.imread(str(self.img_files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        t = self.indices[i]
        frame_ids = range(t - WINDOW + 1, t + 1)
        images = torch.stack([self._load_frame(j) for j in frame_ids])                  
        fsr = torch.from_numpy(self.fsr_feats[t - WINDOW + 1 : t + 1])                
        safety = torch.from_numpy(self.safety[t - WINDOW + 1 : t + 1])              
        target = torch.from_numpy(self.alpha[t : t + self.chunk_size])                
        return {"images": images, "fsr": fsr, "safety": safety, "target": target}

def paper_split(
    train_dirs: Sequence[str],
    valtest_dir: str,
    *,
    chunk_size: int = CHUNK,
    img_size: int = IMG_SIZE,
) -> tuple[Dataset, Dataset, Dataset]:

    kw = {"chunk_size": chunk_size, "img_size": img_size}
    train = ConcatDataset([AuthorityDataset(d, **kw) for d in train_dirs])
    probe = AuthorityDataset(valtest_dir, **kw)
    mid = probe.n // 2
    val = AuthorityDataset(valtest_dir, frame_range=(0, mid), **kw)
    test = AuthorityDataset(valtest_dir, frame_range=(mid, probe.n), **kw)
    return train, val, test

def temporal_split(
    root: str,
    val_ratio: float = 0.2,
    test_ratio: Optional[float] = None,
    *,
    chunk_size: int = CHUNK,
    img_size: int = IMG_SIZE,
) -> tuple[Dataset, Dataset, Dataset]:

    val_ratio = float(val_ratio)
    test_ratio = val_ratio if test_ratio is None else float(test_ratio)
    if not np.isfinite(val_ratio) or not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be finite and strictly between 0 and 1")
    if not np.isfinite(test_ratio) or not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be finite and strictly between 0 and 1")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be strictly less than 1")

    kw = {"chunk_size": chunk_size, "img_size": img_size}
    probe = AuthorityDataset(root, **kw)
    val_count = int(probe.n * val_ratio)
    test_count = int(probe.n * test_ratio)
    train_end = probe.n - val_count - test_count
    val_end = probe.n - test_count
    ranges = ((0, train_end), (train_end, val_end), (val_end, probe.n))

    min_frames = WINDOW + chunk_size - 1
    if any(hi - lo < min_frames for lo, hi in ranges):
        raise ValueError(
            "recording is too short for non-overlapping train/validation/test splits: "
            f"n={probe.n}, val_ratio={val_ratio}, test_ratio={test_ratio}, "
            f"chunk_size={chunk_size}; each split needs at least {min_frames} frames"
        )

    train, val, test = (
        AuthorityDataset(root, frame_range=frame_range, **kw)
        for frame_range in ranges
    )
    return train, val, test

class _ChunkHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.4, chunk_size: int = CHUNK):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2 * self.chunk_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).view(-1, self.chunk_size, 2)
        return ALPHA_MAX * torch.sigmoid(logits)

class _VisualOnly(nn.Module):

    def forward(self, images, fsr=None, safety=None):                     
        return self.head(self.encode(images))

class ThreeLayerCNN(_VisualOnly):

    def __init__(self, dropout: float = 0.4, pretrained: bool = True, chunk_size: int = CHUNK):                
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=2, padding=1),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
            )
        self.conv = nn.Sequential(block(3 * WINDOW, 32), block(32, 64), block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = _ChunkHead(128, dropout, chunk_size)

    def encode(self, images):
        B, T, C, H, W = images.shape
        x = self.conv(images.reshape(B, T * C, H, W))
        return self.pool(x).flatten(1)

class _FrameBackbone(_VisualOnly):

    def __init__(self, backbone: nn.Module, feat_dim: int, dropout: float = 0.4, chunk_size: int = CHUNK):
        super().__init__()
        self.backbone = backbone
        self.head = _ChunkHead(feat_dim, dropout, chunk_size)

    def encode(self, images):
        B, T, C, H, W = images.shape
        feats = self.backbone(images.reshape(B * T, C, H, W)).view(B, T, -1)
        return feats.mean(dim=1)

def _resnet18_frame(dropout: float = 0.4, pretrained: bool = True, chunk_size: int = CHUNK) -> _FrameBackbone:
    w = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = tv_models.resnet18(weights=w)
    m.fc = nn.Identity()
    return _FrameBackbone(m, 512, dropout, chunk_size)

def _resnet50_frame(dropout: float = 0.4, pretrained: bool = True, chunk_size: int = CHUNK) -> _FrameBackbone:
    w = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = tv_models.resnet50(weights=w)
    m.fc = nn.Identity()
    return _FrameBackbone(m, 2048, dropout, chunk_size)

class TemporalShift(nn.Module):

    def __init__(self, n_segment: int = WINDOW, shift_div: int = 8):
        super().__init__()
        self.n_segment, self.shift_div = n_segment, shift_div

    def forward(self, x):
        bt, c, h, w = x.shape
        b, t = bt // self.n_segment, self.n_segment
        ns = c // self.shift_div
        x = x.view(b, t, c, h, w)
        out = x.clone()
        out[:, 1:, :ns] = x[:, :-1, :ns]                                      
        out[:, :-1, ns : 2 * ns] = x[:, 1:, ns : 2 * ns]                       
        return out.view(bt, c, h, w)

def _tsm_resnet18(dropout: float = 0.4, pretrained: bool = True, chunk_size: int = CHUNK) -> _FrameBackbone:
    w = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = tv_models.resnet18(weights=w)
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        for block in getattr(m, layer_name):
            block.conv1 = nn.Sequential(TemporalShift(WINDOW), block.conv1)
    m.fc = nn.Identity()
    return _FrameBackbone(m, 512, dropout, chunk_size)

class R2Plus1D(_VisualOnly):
    def __init__(self, dropout: float = 0.4, pretrained: bool = True, chunk_size: int = CHUNK):                
        super().__init__()
        m = tv_video.r2plus1d_18(weights=None)
        m.fc = nn.Identity()
        self.backbone = m
        self.head = _ChunkHead(512, dropout, chunk_size)

    def encode(self, images):
        return self.backbone(images.permute(0, 2, 1, 3, 4))                 

class MViT(_VisualOnly):

    def __init__(self, dropout: float = 0.4, pretrained: bool = True, chunk_size: int = CHUNK):                
        super().__init__()
        m = tv_video.mvit_v2_s(weights=None)
        m.head = nn.Identity()
        self.backbone = m
        self.head = _ChunkHead(768, dropout, chunk_size)

    def encode(self, images):
        x = images.permute(0, 2, 1, 3, 4)                                   
        x = F.interpolate(x, size=(16, x.shape[-2], x.shape[-1]), mode="trilinear",
                          align_corners=False)
        return self.backbone(x)

class VisionTransformer(_VisualOnly):

    def __init__(self, dropout: float = 0.4, pretrained: bool = True, chunk_size: int = CHUNK):
        super().__init__()
        w = tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        m = tv_models.vit_b_16(weights=w)
        m.heads = nn.Identity()
        self.backbone = m
        self.head = _ChunkHead(768, dropout, chunk_size)

    def encode(self, images):
        B, T, C, H, W = images.shape
        feats = self.backbone(images.reshape(B * T, C, H, W)).view(B, T, -1)
        return feats.mean(dim=1)

MODEL_REGISTRY = {
    "three_layer_cnn": ThreeLayerCNN,
    "resnet18": _resnet18_frame,
    "resnet50": _resnet50_frame,
    "tsm_resnet18": _tsm_resnet18,
    "r2plus1d": R2Plus1D,
    "mvit": MViT,
    "vit": VisionTransformer,
    "bicast": BiCAST,
}

def build_model(
    model_name: str,
    *,
    chunk_size: int = CHUNK,
    pretrained: bool = True,
    bicast_backbone: str = "resnet50",
    input_modality: str = "vision_fsr_safety",
) -> nn.Module:

    if model_name != "bicast":
        return MODEL_REGISTRY[model_name](pretrained=pretrained, chunk_size=chunk_size)

    modalities = {
        "vision": (False, False),
        "vision_fsr": (True, False),
        "vision_fsr_safety": (True, True),
    }
    if input_modality not in modalities:
        raise ValueError(f"unknown input_modality: {input_modality}")
    use_fsr, use_safety = modalities[input_modality]
    return BiCAST(
        chunk_size=chunk_size,
        pretrained_backbone=pretrained,
        backbone_name=bicast_backbone,
        use_fsr=use_fsr,
        use_safety=use_safety,
    )

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:

    model.eval()
    preds, tgts = [], []
    aggregator: Optional[ChunkAggregator] = None

    for batch in loader:
        out = model(
            batch["images"].to(device),
            batch["fsr"].to(device),
            batch["safety"].to(device),
        )

        if aggregator is None:
            aggregator = ChunkAggregator(chunk_size=int(out.shape[1]))
        elif int(out.shape[1]) != aggregator.chunk_size:
            raise ValueError(
                "model chunk size changed during evaluation: "
                f"expected {aggregator.chunk_size}, got {int(out.shape[1])}"
            )

        for b in range(out.shape[0]):
            preds.append(aggregator.push(out[b]).cpu())
            tgts.append(batch["target"][b, 0].cpu())

    if not preds:
        raise ValueError("cannot evaluate an empty temporal dataset")

    p = torch.stack(preds).numpy()                                        
    t = torch.stack(tgts).numpy()                                         

    if len(p) > 1:
        smooth = np.abs(np.diff(p, axis=0)).mean(axis=0)               
    else:
        smooth = np.zeros(2, dtype=np.float32)

    def arm(j):
        err = p[:, j] - t[:, j]
        mae = float(np.abs(err).mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        ss_res = float((err ** 2).sum())
        ss_tot = float(((t[:, j] - t[:, j].mean()) ** 2).sum()) + 1e-12
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": 1.0 - ss_res / ss_tot,
            "smoothness": float(smooth[j]),
        }

    L, R = arm(0), arm(1)
    return {
        "L": L,
        "R": R,
        "smoothness": {
            "L": L["smoothness"],
            "R": R["smoothness"],
            "mean": 0.5 * (L["smoothness"] + R["smoothness"]),
        },
    }

def train_one(
    model_name: str,
    datasets: tuple[Dataset, Dataset, Dataset],
    out_dir: Path,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    lambda_s: float = 0.1,
    lambda_1: float = 1e-3,
    num_workers: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    chunk_size: int = CHUNK,
    bicast_backbone: str = "resnet50",
    input_modality: str = "vision_fsr_safety",
    pretrained: bool = True,
) -> Dict:
    train_set, val_set, test_set = datasets
    dl = lambda ds, sh: DataLoader(
        ds, batch_size=batch_size, shuffle=sh, num_workers=num_workers, pin_memory=True
    )
    train_loader, val_loader, test_loader = dl(train_set, True), dl(val_set, False), dl(test_set, False)

    device_obj = torch.device(device)
    model = build_model(
        model_name,
        chunk_size=chunk_size,
        pretrained=pretrained,
        bicast_backbone=bicast_backbone,
        input_modality=input_modality,
    ).to(device_obj)
    criterion = BiCASTLoss(lambda_s=lambda_s, lambda_1=lambda_1)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_mae, best_path = float("inf"), out_dir / f"model_{model_name}.pth"
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for batch in train_loader:
            optim.zero_grad(set_to_none=True)
            pred = model(
                batch["images"].to(device_obj),
                batch["fsr"].to(device_obj),
                batch["safety"].to(device_obj),
            )
            loss = criterion(pred, batch["target"].to(device_obj), model)
            loss.backward()
            optim.step()
            running += float(loss.detach())
        sched.step()

        val_metrics = evaluate(model, val_loader, device_obj)
        val_mae = 0.5 * (val_metrics["L"]["mae"] + val_metrics["R"]["mae"])
        history.append({"epoch": epoch, "train_loss": running / max(len(train_loader), 1),
                        "val": val_metrics})
        print(f"[{model_name}] epoch {epoch:3d}/{epochs} "
              f"loss={history[-1]['train_loss']:.4f} "
              f"val MAE L/R={val_metrics['L']['mae']:.4f}/{val_metrics['R']['mae']:.4f} "
              f"({time.time() - t0:.1f}s)")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device_obj))
    test_metrics = evaluate(model, test_loader, device_obj)
    result = {
        "model": model_name,
        "best_val_mae": best_mae,
        "test": test_metrics,
        "config": {
            "chunk_size": int(chunk_size),
            "lambda_s": float(lambda_s),
            "lambda_1": float(lambda_1),
            "bicast_backbone": bicast_backbone if model_name == "bicast" else None,
            "input_modality": input_modality if model_name == "bicast" else "vision",
        },
    }
    with open(out_dir / f"history_{model_name}.json", "w") as f:
        json.dump({"history": history, "result": result}, f, indent=2)
    print(f"[{model_name}] TEST  "
          f"MAE L/R={test_metrics['L']['mae']:.4f}/{test_metrics['R']['mae']:.4f}  "
          f"R2 L/R={test_metrics['L']['r2']:.4f}/{test_metrics['R']['r2']:.4f}  "
          f"Smooth L/R={test_metrics['L']['smoothness']:.4f}/"
          f"{test_metrics['R']['smoothness']:.4f}")
    return result

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("data (choose one)")
    src.add_argument("--train_dirs", nargs="+", default=None,
                     help="paper split: Task 1 & Task 3 recording dirs")
    src.add_argument("--valtest_dir", default=None,
                     help="paper split: Task 2 recording dir (-> val/test)")
    src.add_argument("--data_root", default=None,
                     help="single recording; chronological train/validation/test split")
    ap.add_argument("--models", nargs="+", default=["bicast"],
                    choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument(
        "--chunk_size", type=int, default=CHUNK, choices=[1, 5, 10],
        help="prediction chunk C; paper Table IV evaluates 1, 5, 10",
    )
    ap.add_argument(
        "--bicast_backbone", default="resnet50",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Bi-CAST visual backbone; paper Table IV compares 18/34/50",
    )
    ap.add_argument(
        "--input_modality", default="vision_fsr_safety",
        choices=["vision", "vision_fsr", "vision_fsr_safety"],
        help="Bi-CAST modality ablation from paper Table IV",
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", "--weight_decay", dest="weight_decay",
                    type=float, default=1e-4)
    ap.add_argument("--lambda_s", type=float, default=0.1)
    ap.add_argument("--lambda_1", type=float, default=1e-3)
    ap.add_argument(
        "--val_ratio", type=float, default=0.2,
        help="fraction of a single recording assigned to validation",
    )
    ap.add_argument(
        "--test_ratio", type=float, default=None,
        help="fraction assigned to test; defaults to --val_ratio",
    )
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--output_dir", default="outputs/authority_models")
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--hidden_dim", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--num_layers", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.train_dirs and args.valtest_dir:
        datasets = paper_split(args.train_dirs, args.valtest_dir, chunk_size=args.chunk_size, img_size=args.img_size)
    elif args.data_root:
        datasets = temporal_split(
            args.data_root,
            args.val_ratio,
            args.test_ratio,
            chunk_size=args.chunk_size,
            img_size=args.img_size,
        )
    else:
        ap.error("provide either --train_dirs + --valtest_dir, or --data_root")

    out_dir = Path(args.output_dir)
    results = [
        train_one(
            name, datasets, out_dir / name,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, lambda_s=args.lambda_s,
            lambda_1=args.lambda_1, num_workers=args.num_workers,
            chunk_size=args.chunk_size, bicast_backbone=args.bicast_backbone,
            input_modality=args.input_modality,
        )
        for name in args.models
    ]
    with open(out_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Results -> {out_dir / 'comparison_results.json'}")

if __name__ == "__main__":
    main()
