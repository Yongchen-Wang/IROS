#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iros.models.bicast import BiCAST, ChunkAggregator              

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiCAST(pretrained_backbone=False).to(device).eval()

    aggregator = ChunkAggregator(chunk_size=model.chunk_size)

    for step in range(8):                                      
        images = torch.rand(1, model.n_frames, 3, 224, 224, device=device)
        fsr = torch.rand(1, model.n_frames, 2, 3, device=device)
        safety = torch.rand(1, model.n_frames, 2, device=device)

        with torch.no_grad():
            chunk = model(images, fsr, safety)[0]                
        alpha = aggregator.push(chunk)                                        

        u_human = torch.rand(2, 3)                                          
        u_robot = torch.rand(2, 3)                                         
        u = alpha[:, None] * u_robot + (1 - alpha)[:, None] * u_human         

        print(f"step {step}: alpha_L={alpha[0]:.3f} alpha_R={alpha[1]:.3f} "
              f"|u_L|={u[0].norm():.3f} |u_R|={u[1].norm():.3f}")

if __name__ == "__main__":
    main()
