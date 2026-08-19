from __future__ import annotations

"""
Export navigation trajectories to CSV for offline analysis.

Run from project root:
    python -m iros.navigation.export_trajectory_csv --dataset DATASET --target TARGET
"""

import argparse
import csv
from math import hypot
from pathlib import Path
from typing import Optional

from .config import NavigationConfig, load_config
from .data_loader import load_navigation_data
from .trajectory_generator import FullTrajectory, generate_full_trajectory
from ..paths import CONFIG_ROOT, OUTPUT_ROOT

def _load_full_trajectory(
    dataset_name: str,
    target_name: str,
    config_path: Optional[str] = None,
) -> FullTrajectory:

    config: NavigationConfig = load_config(
        config_path or str(CONFIG_ROOT / "navigation" / "default.json")
    )

    print(f"[CSV] Loading dataset: {dataset_name}")
    data = load_navigation_data(dataset_name, config)

    print(f"[CSV] Generating trajectory to target '{target_name}'")
    traj = generate_full_trajectory(data, config, target_name=target_name)
    return traj

def export_trajectory_to_csv(
    dataset_name: str,
    target_name: str = "A",
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:

    traj = _load_full_trajectory(
        dataset_name=dataset_name,
        target_name=target_name,
        config_path=config_path,
    )

    waypoints = traj.waypoints
    if not waypoints:
        raise ValueError("No waypoints found in trajectory; cannot export CSV.")

    goal_y, goal_x = traj.path_result.goal_maze

    if output_path is None:
        default_dir = OUTPUT_ROOT / "navigation"
        default_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(
            default_dir / f"trajectory_{dataset_name}_{target_name}.csv"
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "index",
        "center_y",
        "center_x",
        "bottom_y",
        "bottom_x",
        "top_y",
        "top_x",
        "closer_endpoint",
        "closer_y",
        "closer_x",
        "farther_endpoint",
        "farther_y",
        "farther_x",
        "tangent_closer_dy",
        "tangent_closer_dx",
        "tangent_farther_dy",
        "tangent_farther_dx",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, wp in enumerate(waypoints):
            cy, cx = wp.center_maze
            by, bx = wp.bottom_maze
            ty, tx = wp.top_maze
            dy, dx = wp.tangent_maze

            bottom_dist = hypot(by - goal_y, bx - goal_x)
            top_dist = hypot(ty - goal_y, tx - goal_x)

            if bottom_dist <= top_dist:
                closer_endpoint = "bottom"
                closer_y, closer_x = by, bx
                farther_endpoint = "top"
                farther_y, farther_x = ty, tx
            else:
                closer_endpoint = "top"
                closer_y, closer_x = ty, tx
                farther_endpoint = "bottom"
                farther_y, farther_x = by, bx

            row = {
                "index": idx,
                "center_y": float(cy),
                "center_x": float(cx),
                "bottom_y": float(by),
                "bottom_x": float(bx),
                "top_y": float(ty),
                "top_x": float(tx),
                "closer_endpoint": closer_endpoint,
                "closer_y": float(closer_y),
                "closer_x": float(closer_x),
                "farther_endpoint": farther_endpoint,
                "farther_y": float(farther_y),
                "farther_x": float(farther_x),
                "tangent_closer_dy": float(dy),
                "tangent_closer_dx": float(dx),
                "tangent_farther_dy": float(dy),
                "tangent_farther_dx": float(dx),
            }
            writer.writerow(row)

    print(f"[CSV] Saved trajectory CSV to: {out_path}")
    return str(out_path)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export navigation trajectory to CSV for a given dataset and target."
    )
    parser.add_argument(
        "--dataset",
        default="robot_data_20260220_195724",
        help="Dataset name under IROS_DATA_ROOT.",
    )
    parser.add_argument(
        "--target",
        default="A",
        help="Target name (e.g. 'A', 'B', 'C').",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Navigation config JSON (defaults to configs/navigation/default.json).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV path (defaults to outputs/navigation/trajectory_<dataset>_<target>.csv).",
    )
    args = parser.parse_args()

    export_trajectory_to_csv(
        dataset_name=args.dataset,
        target_name=args.target,
        config_path=args.config,
        output_path=args.output,
    )

if __name__ == "__main__":
    main()
