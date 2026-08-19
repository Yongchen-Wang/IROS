from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib

if not os.getenv("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .arm_switching import ArmRole
from .config import NavigationConfig, load_config
from .costmap_builder import build_safety_costmap
from .coordinate_transform import maze_to_cam
from .data_loader import load_navigation_data
from .trajectory_generator import generate_full_trajectory
from ..paths import CONFIG_ROOT, OUTPUT_ROOT, PROJECT_ROOT

def _load_maze_image(maze_params: dict, project_root: Path) -> Optional[np.ndarray]:

    maze_path = maze_params.get("maze_path")
    if not maze_path:
        return None
    p = Path(maze_path)
    if not p.is_absolute():
        p = project_root / p
    if not p.exists():
        return None
    try:
        from skimage import io
        img = io.imread(str(p))
        if img.ndim == 3:
            return img
        return np.stack([img] * 3, axis=-1)
    except Exception:
        return None

def visualize_with_costmap_overlay(
    dataset_name: str = "robot_data_20260220_195724",
    target_name: str = "A",
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = True,
    costmap_alpha: float = 0.5,
    label_fontsize: int = 14,
    legend_fontsize: int = 12,
    title_fontsize: int = 16,
) -> None:

    project_root = PROJECT_ROOT

    config = load_config(config_path or str(CONFIG_ROOT / "navigation" / "default.json"))

    print(f"Loading dataset: {dataset_name}")
    data = load_navigation_data(dataset_name, config)

    print(f"Generating trajectory to target '{target_name}'")
    traj = generate_full_trajectory(data, config, target_name=target_name)

    maze_shape = traj.maze_shape
    maze_params = traj.maze_params
    waypoints = traj.waypoints
    switch_indices = set(traj.switch_indices)

    costmap = build_safety_costmap(
        maze_extractor=data.maze_extractor,
        robot_radius_px=data.robot_radius_px,
        config=config,
        robot_length_px=data.ump_rod_length_maze,
    )

    maze_img = _load_maze_image(maze_params, project_root)
    if maze_img is not None:

        bg = maze_img
        extent = [0, maze_shape[1], maze_shape[0], 0]                                               
    else:
        free_mask = np.asarray(data.maze_extractor.free_mask)
        bg = np.where(free_mask[:, :, np.newaxis], [0.95, 0.95, 0.9], [0.3, 0.3, 0.3])
        extent = [0, maze_shape[1], maze_shape[0], 0]

    costs_display = np.ma.masked_where(~costmap.valid_mask, costmap.costs)
    costs_display = np.ma.masked_where(costs_display == np.inf, costs_display)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))

    ax1.imshow(bg, extent=extent, origin="upper", aspect="auto")

    im1 = ax1.imshow(costs_display, cmap="YlGnBu", alpha=0.6,
                     extent=extent, origin="upper", aspect="auto", vmin=0, vmax=5)
    ax1.set_xlim(0, maze_shape[1])
    ax1.set_ylim(maze_shape[0], 0)
    ax1.set_aspect("equal")
    ax1.set_title(f"Path Planning (Target {target_name})",
                  fontsize=title_fontsize)
    ax1.set_xlabel("X", fontsize=label_fontsize)
    ax1.set_ylabel("Y", fontsize=label_fontsize)

    path_y = [wp.center_maze[0] for wp in waypoints]
    path_x = [wp.center_maze[1] for wp in waypoints]
    ax1.plot(path_x, path_y, "k-", linewidth=2.5, zorder=2)

    for i, wp in enumerate(waypoints):
        color = "C0" if wp.master == ArmRole.BOTTOM else "C1"
        ax1.scatter(wp.center_maze[1], wp.center_maze[0], c=color, s=12, alpha=0.8, zorder=3)

    start_idx = 1
    bottom_y = [wp.bottom_maze[0] for wp in waypoints[start_idx:-2]]
    bottom_x = [wp.bottom_maze[1] for wp in waypoints[start_idx:-2]]
    top_y = [wp.top_maze[0] for wp in waypoints[start_idx:-2]]
    top_x = [wp.top_maze[1] for wp in waypoints[start_idx:-2]]
    ax1.plot(bottom_x, bottom_y, "b--", linewidth=1.2, alpha=0.7, label="Bottom endpoint")
    ax1.plot(top_x, top_y, color="orange", linestyle="--", linewidth=1.2, alpha=0.7, label="Top endpoint")

    ump_rod_width = data.ump_rod_width_maze if data.ump_rod_width_maze else 10.0
    ump_rod_half_width = ump_rod_width / 2.0

    sorted_switch_indices = sorted(switch_indices)

    all_bottom_y = [wp.bottom_maze[0] for wp in waypoints]
    all_bottom_x = [wp.bottom_maze[1] for wp in waypoints]
    all_top_y = [wp.top_maze[0] for wp in waypoints]
    all_top_x = [wp.top_maze[1] for wp in waypoints]
    all_masters = [wp.master for wp in waypoints]

    segments = []
    if len(waypoints) > 1:
        current_role = waypoints[0].master
        seg_start = 0

        for i in range(1, len(waypoints)):
            if waypoints[i].master != current_role:

                if i >= 1:
                    seg_start = i
                    current_role = waypoints[i].master
                    break

        if len(waypoints) - seg_start >= 2:
            segments.append((seg_start, len(waypoints)))

    from matplotlib.patches import Polygon

    for seg_start, seg_end in segments:

        if seg_end - seg_start < 2:
            continue

        seg_role = all_masters[seg_start]

        seg_bottom_x = all_bottom_x[seg_start:seg_end-1]
        seg_bottom_y = all_bottom_y[seg_start:seg_end-1]
        seg_top_x = all_top_x[seg_start:seg_end-1]
        seg_top_y = all_top_y[seg_start:seg_end-1]

        if seg_role == ArmRole.BOTTOM:

            left_wall = 0
            polygon_x = [left_wall]
            polygon_y = [seg_bottom_y[0]]
            for bx, by in zip(seg_bottom_x, seg_bottom_y):
                polygon_x.extend([bx, bx])
                polygon_y.extend([by - ump_rod_half_width, by + ump_rod_half_width])
            polygon_x.append(left_wall)
            polygon_y.append(seg_bottom_y[-1])

            patch = Polygon(
                list(zip(polygon_x, polygon_y)),
                closed=True,
                facecolor="red",
                edgecolor="darkred",
                alpha=0.15,
                linewidth=1,
                zorder=1
            )
        else:            

            right_wall = maze_shape[1]
            polygon_x = [right_wall]
            polygon_y = [seg_top_y[0]]
            for tx, ty in zip(seg_top_x, seg_top_y):
                polygon_x.extend([tx, tx])
                polygon_y.extend([ty - ump_rod_half_width, ty + ump_rod_half_width])
            polygon_x.append(right_wall)
            polygon_y.append(seg_top_y[-1])

            patch = Polygon(
                list(zip(polygon_x, polygon_y)),
                closed=True,
                facecolor="blue",
                edgecolor="darkblue",
                alpha=0.15,
                linewidth=1,
                zorder=1
            )

        ax1.add_patch(patch)

    robot_length = data.ump_rod_length_maze / 2.0 if data.ump_rod_length_maze else 58.5

    key_indices = []

    if len(waypoints) > 2:
        key_indices.append(1)                      

    key_indices.append(len(waypoints) // 2)

    if len(waypoints) > 3:
        key_indices.append(len(waypoints) - 3)                 

    for si in sorted(switch_indices):
        if si > 2 and si < len(waypoints) - 3:
            key_indices.append(si)

    key_indices = sorted(set(key_indices))

    first_key = True                            
    for i in key_indices:
        wp = waypoints[i]

        if first_key:

            dx, dy = 0, 1
            first_key = False
        else:

            prev_wp = waypoints[max(0, i - 1)]
            next_wp = waypoints[min(len(waypoints) - 1, i + 1)]
            dx = next_wp.center_maze[1] - prev_wp.center_maze[1]
            dy = next_wp.center_maze[0] - prev_wp.center_maze[0]

            mag = np.sqrt(dx * dx + dy * dy)
            if mag > 0:
                dx, dy = dx / mag, dy / mag

        center_x, center_y = wp.center_maze[1], wp.center_maze[0]
        half_len = robot_length

        bottom_x_rot = center_x - dx * half_len
        bottom_y_rot = center_y - dy * half_len
        top_x_rot = center_x + dx * half_len
        top_y_rot = center_y + dy * half_len

        ax1.plot(
            [bottom_x_rot, top_x_rot],
            [bottom_y_rot, top_y_rot],
            "g-", linewidth=3, alpha=0.8, zorder=4
        )

        ax1.scatter(bottom_x_rot, bottom_y_rot, c="blue", s=50, alpha=0.9, zorder=5, edgecolors="black", linewidths=0.5)

        ax1.scatter(top_x_rot, top_y_rot, c="orange", s=50, alpha=0.9, zorder=5, edgecolors="black", linewidths=0.5)

        ax1.scatter(center_x, center_y, c="black", s=30, alpha=0.9, zorder=5, marker="o")

    start = waypoints[0]
    goal = waypoints[-1]
    ax1.scatter(start.center_maze[1], start.center_maze[0], c="green", s=150, marker="s",
               label="Start", zorder=5, edgecolors="black", linewidths=1)
    ax1.scatter(goal.center_maze[1], goal.center_maze[0], c="red", s=200, marker="*",
               label="Goal", zorder=5, edgecolors="black", linewidths=1)

    ax1.legend(loc="lower left", fontsize=legend_fontsize, framealpha=0.9)
    ax1.tick_params(axis='both', which='major', labelsize=label_fontsize - 2)
    ax1.grid(True, alpha=0.3, linestyle='--')

    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    cbar1.set_label("Travel Cost", fontsize=label_fontsize - 2)
    cbar1.ax.tick_params(labelsize=label_fontsize - 2)

    plt.tight_layout()

    if not output_path:
        output_dir = OUTPUT_ROOT / "navigation"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"visualization_costmap_{dataset_name}_{target_name}.png")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if show:
        try:

            import matplotlib
            if matplotlib.get_backend() == "Agg" and os.getenv("DISPLAY"):
                matplotlib.use("TkAgg")
            plt.show()
        except Exception as e:
            print(f"Note: Could not display figure interactively ({e}). Figure saved to file.")
    else:
        plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize navigation trajectory with costmap overlay"
    )
    parser.add_argument(
        "--dataset", default="robot_data_20260220_195724",
        help="Dataset name"
    )
    parser.add_argument(
        "--target", default="A",
        help="Target name (A, B, C)"
    )
    parser.add_argument(
        "--config", default=None,
        help="Config JSON path"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output figure path"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not show interactive window"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Costmap transparency (0-1), lower = more transparent"
    )
    args = parser.parse_args()

    visualize_with_costmap_overlay(
        dataset_name=args.dataset,
        target_name=args.target,
        config_path=args.config,
        output_path=args.output,
        show=not args.no_show,
        costmap_alpha=args.alpha,
    )

if __name__ == "__main__":
    main()
