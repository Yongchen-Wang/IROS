#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA_MIN, ALPHA_MAX = 0.0, 0.9                                   

def load_annotation(path: Path) -> np.ndarray:

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        need = ["frame", "alpha_l", "alpha_r"]
        missing = [c for c in need if c not in cols]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        df = df.sort_values(cols["frame"])
        return df[[cols["alpha_l"], cols["alpha_r"]]].to_numpy(dtype=np.float64)
    with open(path, "rb") as f:
        d = pickle.load(f)
    return np.stack([np.asarray(d["alpha_L"]), np.asarray(d["alpha_R"])], axis=1)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", nargs="+", required=True,
                    help="per-annotator score files (CSV or PKL)")
    ap.add_argument("--recording", required=True,
                    help="recording directory; writes <recording>/alpha_labels.pkl")
    ap.add_argument("--min_annotators", type=int, default=3,
                    help="require at least this many annotators (paper: 3)")
    args = ap.parse_args()

    files = [Path(p) for p in args.annotations]
    if len(files) < args.min_annotators:
        raise SystemExit(
            f"got {len(files)} annotation files, need >= {args.min_annotators} "
            f"(the paper protocol uses three independent annotators)"
        )

    scores = [load_annotation(p) for p in files]
    n = min(s.shape[0] for s in scores)
    if any(s.shape[0] != n for s in scores):
        print(f"note: annotators cover different lengths; truncating to {n} frames")
    stack = np.stack([s[:n] for s in scores])                                

    merged = np.clip(stack.mean(axis=0), ALPHA_MIN, ALPHA_MAX)             

    A = stack.shape[0]
    pair_mad = np.mean(
        [np.abs(stack[a] - stack[b]).mean() for a in range(A) for b in range(a + 1, A)]
    )
    print(f"annotators: {A}, frames: {n}, mean pairwise |diff|: {pair_mad:.4f}")

    out = Path(args.recording) / "alpha_labels.pkl"
    labels: dict = {}
    if out.exists():
        with open(out, "rb") as f:
            labels = pickle.load(f)
        print(f"updating existing {out} (keys preserved: "
              f"{sorted(k for k in labels if k not in ('alpha_L', 'alpha_R'))})")
    labels["alpha_L"] = merged[:, 0].astype(np.float32)
    labels["alpha_R"] = merged[:, 1].astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(labels, f)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
