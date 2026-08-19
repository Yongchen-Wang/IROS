#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.core.train_model import (
    CHUNK,
    IMG_SIZE,
    paper_split,
    temporal_split,
    train_one,
)

RUNS = {
    "backbone_resnet18": {
        "bicast_backbone": "resnet18", "input_modality": "vision_fsr_safety",
        "lambda_s": 0.1, "chunk_size": 5,
    },
    "backbone_resnet34": {
        "bicast_backbone": "resnet34", "input_modality": "vision_fsr_safety",
        "lambda_s": 0.1, "chunk_size": 5,
    },
    "reference": {
        "bicast_backbone": "resnet50", "input_modality": "vision_fsr_safety",
        "lambda_s": 0.1, "chunk_size": 5,
    },
    "modality_vision": {
        "bicast_backbone": "resnet50", "input_modality": "vision",
        "lambda_s": 0.1, "chunk_size": 5,
    },
    "modality_vision_fsr": {
        "bicast_backbone": "resnet50", "input_modality": "vision_fsr",
        "lambda_s": 0.1, "chunk_size": 5,
    },
    "smooth_lambda0": {
        "bicast_backbone": "resnet50", "input_modality": "vision_fsr_safety",
        "lambda_s": 0.0, "chunk_size": 5,
    },
    "chunk_c1": {
        "bicast_backbone": "resnet50", "input_modality": "vision_fsr_safety",
        "lambda_s": 0.1, "chunk_size": 1,
    },
    "chunk_c10": {
        "bicast_backbone": "resnet50", "input_modality": "vision_fsr_safety",
        "lambda_s": 0.1, "chunk_size": 10,
    },
}

TABLE_ROWS = [
    ("Visual Backbone", "ResNet-18", "backbone_resnet18"),
    ("Visual Backbone", "ResNet-34", "backbone_resnet34"),
    ("Visual Backbone", "ResNet-50", "reference"),
    ("Input Modality", "Vision only", "modality_vision"),
    ("Input Modality", "Vision + FSR", "modality_vision_fsr"),
    ("Input Modality", "Vision + FSR + Safety", "reference"),
    ("Smoothness", "lambda_s = 0", "smooth_lambda0"),
    ("Chunk Size", "C = 1", "chunk_c1"),
    ("Chunk Size", "C = 5", "reference"),
    ("Chunk Size", "C = 10", "chunk_c10"),
]

def bilateral_average(result: dict) -> dict:

    test = result["test"]
    return {
        "mae": 0.5 * (test["L"]["mae"] + test["R"]["mae"]),
        "rmse": 0.5 * (test["L"]["rmse"] + test["R"]["rmse"]),
        "r2": 0.5 * (test["L"]["r2"] + test["R"]["r2"]),
    }

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_argument_group("data (choose one)")
    src.add_argument("--train_dirs", nargs="+", default=None)
    src.add_argument("--valtest_dir", default=None)
    src.add_argument("--data_root", default=None)
    ap.add_argument("--only", nargs="+", choices=list(RUNS), default=None,
                    help="run only selected unique configurations")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", "--wd", type=float, default=1e-4)
    ap.add_argument("--lambda_1", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument(
        "--val_ratio", type=float, default=0.2,
        help="fraction of a single recording assigned to validation",
    )
    ap.add_argument(
        "--test_ratio", type=float, default=None,
        help="fraction assigned to test; defaults to --val_ratio",
    )
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_dir", default="outputs/table_iv_ablations")
    ap.add_argument("--no_pretrained", action="store_true",
                    help="offline/debug only; paper uses ImageNet-pretrained backbones")
    args = ap.parse_args()

    if not ((args.train_dirs and args.valtest_dir) or args.data_root):
        ap.error("provide --train_dirs + --valtest_dir, or --data_root")

    selected = args.only or list(RUNS)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    unique_results = {}

    for key in selected:
        cfg = RUNS[key]
        c = int(cfg["chunk_size"])
        print("\n" + "=" * 78)
        print(f"Table IV run: {key}  {cfg}")
        print("=" * 78)

        if args.train_dirs and args.valtest_dir:
            datasets = paper_split(
                args.train_dirs, args.valtest_dir,
                chunk_size=c, img_size=args.img_size,
            )
        else:
            datasets = temporal_split(
                args.data_root, args.val_ratio, args.test_ratio,
                chunk_size=c, img_size=args.img_size,
            )

        result = train_one(
            "bicast",
            datasets,
            out_root / key,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lambda_s=float(cfg["lambda_s"]),
            lambda_1=args.lambda_1,
            num_workers=args.num_workers,
            device=args.device,
            chunk_size=c,
            bicast_backbone=str(cfg["bicast_backbone"]),
            input_modality=str(cfg["input_modality"]),
            pretrained=not args.no_pretrained,
        )
        unique_results[key] = {
            "config": cfg,
            "bilateral_average": bilateral_average(result),
            "full_result": result,
        }

        with open(out_root / "unique_results.json", "w") as f:
            json.dump(unique_results, f, indent=2)

    table = []
    for group, label, key in TABLE_ROWS:
        row = {"group": group, "configuration": label, "run_key": key}
        if key in unique_results:
            row.update(unique_results[key]["bilateral_average"])
        table.append(row)

    payload = {
        "paper_table": "Table IV: Analysis of key design components",
        "metric_definition": "bilateral mean of left/right test MAE, RMSE, and R^2",
        "unique_runs": unique_results,
        "rows_in_paper_order": table,
    }
    out_json = out_root / "table_iv_results.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nTable IV rows available from this run:")
    for row in table:
        if "mae" in row:
            print(
                f"{row['group']:15s} | {row['configuration']:24s} | "
                f"MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} R2={row['r2']:.4f}"
            )
    print(f"\nwrote {out_json}")

if __name__ == "__main__":
    main()
