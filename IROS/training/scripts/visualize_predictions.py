#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt              
import torch              
from torch.utils.data import DataLoader              

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iros.models.bicast import ChunkAggregator              
from training.core.train_model import (              
    CHUNK, AuthorityDataset, MODEL_REGISTRY, build_model,
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--model", default="bicast", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--out", default="alpha_curves.png")
    ap.add_argument("--chunk_size", type=int, default=CHUNK, choices=[1, 5, 10])
    ap.add_argument("--bicast_backbone", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--input_modality", default="vision_fsr_safety", choices=["vision", "vision_fsr", "vision_fsr_safety"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = build_model(
        args.model, chunk_size=args.chunk_size,
        bicast_backbone=args.bicast_backbone,
        input_modality=args.input_modality,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    ds = AuthorityDataset(args.recording, chunk_size=args.chunk_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False)                  
    agg = ChunkAggregator(chunk_size=args.chunk_size)

    executed, labels = [], []
    with torch.no_grad():
        for batch in loader:
            chunk = model(
                batch["images"].to(device),
                batch["fsr"].to(device),
                batch["safety"].to(device),
            )[0]                                          
            executed.append(agg.push(chunk).cpu())
            labels.append(batch["target"][0, 0])                              
    pred = torch.stack(executed).numpy()
    gt = torch.stack(labels).numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for j, (ax, arm) in enumerate(zip(axes, ("Left arm", "Right arm"))):
        ax.plot(gt[:, j], color="tab:blue", label="ground truth")
        ax.plot(pred[:, j], color="tab:red", label="predicted (Eq. 8 aggregated)")
        ax.set_ylabel(rf"$\alpha$ ({arm})")
        ax.set_ylim(-0.02, 0.95)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"saved {args.out}  ({len(pred)} frames)")

if __name__ == "__main__":
    main()
