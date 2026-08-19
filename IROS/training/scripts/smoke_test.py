#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iros.models.bicast import BiCASTLoss, ChunkAggregator              
from training.core.train_model import (              
    ALPHA_MAX, CHUNK, IMG_SIZE, MODEL_REGISTRY, WINDOW, build_model,
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--chunk_size", type=int, default=CHUNK, choices=[1, 5, 10])
    ap.add_argument("--bicast_backbone", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--input_modality", default="vision_fsr_safety", choices=["vision", "vision_fsr", "vision_fsr_safety"])
    args = ap.parse_args()

    B = args.batch
    images = torch.rand(B, WINDOW, 3, IMG_SIZE, IMG_SIZE)
    fsr = torch.rand(B, WINDOW, 2, 3)
    safety = torch.rand(B, WINDOW, 2)
    target = torch.rand(B, args.chunk_size, 2) * ALPHA_MAX
    criterion = BiCASTLoss()

    for name in args.models:
        model = build_model(
            name, chunk_size=args.chunk_size, pretrained=False,
            bicast_backbone=args.bicast_backbone,
            input_modality=args.input_modality,
        )                    
        model.eval()
        with torch.no_grad():
            out = model(images, fsr, safety)
        assert out.shape == (B, args.chunk_size, 2), f"{name}: bad shape {tuple(out.shape)}"
        assert float(out.min()) >= 0.0 and float(out.max()) <= ALPHA_MAX + 1e-6, \
            f"{name}: outputs outside [0, {ALPHA_MAX}]"
        with torch.no_grad():
            loss = criterion(out, target, model)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[ok] {name:16s} out={tuple(out.shape)} "
              f"loss={float(loss):.3f} params={n_params:.1f}M")

    agg = ChunkAggregator(chunk_size=args.chunk_size)
    const = torch.full((args.chunk_size, 2), 0.4)
    for _ in range(args.chunk_size + 2):
        alpha = agg.push(const)
    assert torch.allclose(alpha, const[0], atol=1e-6), "aggregator not consistent"
    print(f"[ok] ChunkAggregator  alpha={alpha.tolist()} (constant-input check)")
    print("all smoke tests passed")

if __name__ == "__main__":
    main()
