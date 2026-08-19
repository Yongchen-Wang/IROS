#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.core.train_model import (              
    CHUNK, AuthorityDataset, MODEL_REGISTRY, build_model, evaluate,
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--half", choices=["full", "first", "second"], default="second",
                    help="paper protocol: Task 2 second half is the test set")
    ap.add_argument("--checkpoint_root", default="outputs/authority_models")
    ap.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--chunk_size", type=int, default=CHUNK, choices=[1, 5, 10])
    ap.add_argument("--bicast_backbone", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--input_modality", default="vision_fsr_safety", choices=["vision", "vision_fsr", "vision_fsr_safety"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="optional JSON results path")
    args = ap.parse_args()

    probe = AuthorityDataset(args.recording, chunk_size=args.chunk_size)
    mid = probe.n // 2
    frame_range = {"full": None, "first": (0, mid), "second": (mid, probe.n)}[args.half]
    ds = AuthorityDataset(args.recording, frame_range=frame_range, chunk_size=args.chunk_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)

    results = {}
    for name in args.models:
        ckpt = Path(args.checkpoint_root) / name / f"model_{name}.pth"
        if not ckpt.exists():
            print(f"[skip] {name}: checkpoint not found at {ckpt}")
            continue
        model = build_model(
            name, chunk_size=args.chunk_size,
            bicast_backbone=args.bicast_backbone,
            input_modality=args.input_modality,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        m = evaluate(model, loader, device)
        results[name] = m
        print(f"{name:16s} MAE L/R={m['L']['mae']:.4f}/{m['R']['mae']:.4f}  "
              f"RMSE L/R={m['L']['rmse']:.4f}/{m['R']['rmse']:.4f}  "
              f"R2 L/R={m['L']['r2']:.4f}/{m['R']['r2']:.4f}  "
              f"Smooth L/R={m['L']['smoothness']:.4f}/{m['R']['smoothness']:.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
