#!/usr/bin/env python3

from __future__ import annotations

import argparse

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="YOLO dataset yaml")
    ap.add_argument("--weights", default="yolov8s-seg.pt",
                    help="COCO-pretrained starting weights")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="e.g. 0 or 0,1 or cpu")
    ap.add_argument("--project", default="outputs/yolo_seg")
    args = ap.parse_args()

    from ultralytics import YOLO                                 

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer="SGD",
        momentum=0.937,
        lr0=0.01,
        lrf=0.1,                                       
        cos_lr=True,
        device=args.device,
        project=args.project,
        name="yolov8s_seg_millirobot",

    )

if __name__ == "__main__":
    main()
