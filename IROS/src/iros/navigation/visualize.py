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

def visualize_navigation(
    dataset_name: str = "robot_data_20260220_195724",
    target_name: str = "A",
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = True,
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

    maze_img = _load_maze_image(maze_params, project_root)
    if maze_img is not None:
        bg = maze_img
        extent = [0, maze_shape[1], maze_shape[0], 0]                                       
    else:
        free_mask = np.asarray(data.maze_extractor.free_mask)
        bg = np.where(free_mask[:, :, np.newaxis], [0.95, 0.95, 0.9], [0.3, 0.3, 0.35])
        extent = [0, maze_shape[1], maze_shape[0], 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    ax1 = axes[0]
    ax1.imshow(bg, extent=extent, origin="upper", aspect="auto")
    ax1.set_xlim(0, maze_shape[1])
    ax1.set_ylim(maze_shape[0], 0)
    ax1.set_aspect("equal")
    ax1.set_title("Maze coordinates: Path & Arm Switching")
    ax1.set_xlabel("Maze X")
    ax1.set_ylabel("Maze Y")

    path_y = [wp.center_maze[0] for wp in waypoints]
    path_x = [wp.center_maze[1] for wp in waypoints]
    ax1.plot(path_x, path_y, "k-", linewidth=2, label="Center path", zorder=2)

    for i, wp in enumerate(waypoints):
        color = "C0" if wp.master == ArmRole.BOTTOM else "C1"
        ax1.scatter(wp.center_maze[1], wp.center_maze[0], c=color, s=8, alpha=0.8, zorder=3)
        if i in switch_indices:
            ax1.scatter(wp.center_maze[1], wp.center_maze[0], c="red", s=80, marker="x", zorder=4)

    bottom_y = [wp.bottom_maze[0] for wp in waypoints]
    bottom_x = [wp.bottom_maze[1] for wp in waypoints]
    top_y = [wp.top_maze[0] for wp in waypoints]
    top_x = [wp.top_maze[1] for wp in waypoints]
    ax1.plot(bottom_x, bottom_y, "b--", linewidth=0.8, alpha=0.6, label="Bottom endpoint")
    ax1.plot(top_x, top_y, "orange", linestyle="--", linewidth=0.8, alpha=0.6, label="Top endpoint")

    start = waypoints[0]
    goal = waypoints[-1]
    ax1.scatter(start.center_maze[1], start.center_maze[0], c="green", s=120, marker="o", label="Start", zorder=5)
    ax1.scatter(goal.center_maze[1], goal.center_maze[0], c="red", s=120, marker="*", label="Goal", zorder=5)

    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    costmap = build_safety_costmap(
        maze_extractor=data.maze_extractor,
        robot_radius_px=data.robot_radius_px,
        config=config,
        robot_length_px=data.ump_rod_length_maze,
    )
    costs_display = np.ma.masked_where(~costmap.valid_mask, costmap.costs)
    costs_display = np.ma.masked_where(costs_display == np.inf, costs_display)
    im = ax2.imshow(costs_display, cmap="viridis", origin="upper", aspect="equal")
    ax2.imshow(~costmap.valid_mask, cmap="gray", alpha=0.4, origin="upper", aspect="equal")
    ax2.plot(path_x, path_y, "w-", linewidth=2, label="Path")
    ax2.scatter(path_x[0], path_y[0], c="lime", s=100, marker="o", label="Start")
    ax2.scatter(path_x[-1], path_y[-1], c="red", s=100, marker="*", label="Goal")
    ax2.set_title("Costmap & Path")
    ax2.set_xlabel("Maze X")
    ax2.set_ylabel("Maze Y")
    ax2.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax2, label="Traversal cost")

    plt.tight_layout()

    if not output_path:
        output_dir = OUTPUT_ROOT / "navigation"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"visualization_{dataset_name}_{target_name}.png")

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

def visualize_arm_trajectories(
    dataset_name: str = "robot_data_20260220_195724",
    target_name: str = "A",
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    show: bool = False,
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

    maze_img = _load_maze_image(maze_params, project_root)
    if maze_img is not None:
        bg = maze_img
        extent = [0, maze_shape[1], maze_shape[0], 0]
    else:
        free_mask = np.asarray(data.maze_extractor.free_mask)
        bg = np.where(free_mask[:, :, np.newaxis], [0.95, 0.95, 0.9], [0.3, 0.3, 0.35])
        extent = [0, maze_shape[1], maze_shape[0], 0]

    path_x = [wp.center_maze[1] for wp in waypoints]
    path_y = [wp.center_maze[0] for wp in waypoints]
    bottom_x = [wp.bottom_maze[1] for wp in waypoints]
    bottom_y = [wp.bottom_maze[0] for wp in waypoints]
    top_x = [wp.top_maze[1] for wp in waypoints]
    top_y = [wp.top_maze[0] for wp in waypoints]

    out_dir = Path(output_dir or str(OUTPUT_ROOT / "navigation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 7))
    ax1.imshow(bg, extent=extent, origin="upper", aspect="auto")
    ax1.set_xlim(0, maze_shape[1])
    ax1.set_ylim(maze_shape[0], 0)
    ax1.set_aspect("equal")
    ax1.set_title("左臂运动轨迹 (Bottom / Left Arm)", fontsize=14)
    ax1.set_xlabel("Maze X")
    ax1.set_ylabel("Maze Y")

    ax1.plot(path_x, path_y, "k-", linewidth=1.5, alpha=0.5, label="Center path", zorder=1)
    ax1.plot(bottom_x, bottom_y, "b-", linewidth=2.5, label="左臂轨迹", zorder=2)
    ax1.scatter(bottom_x, bottom_y, c="C0", s=20, alpha=0.8, zorder=3)
    for i in switch_indices:
        if waypoints[i].master == ArmRole.BOTTOM:
            ax1.scatter(bottom_x[i], bottom_y[i], c="red", s=100, marker="x", zorder=4)
    ax1.scatter(path_x[0], path_y[0], c="green", s=150, marker="o", label="Start", zorder=5)
    ax1.scatter(bottom_x[-1], bottom_y[-1], c="red", s=150, marker="*", label="Goal", zorder=5)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    left_path = out_dir / "visualization_left_arm.png"
    fig1.savefig(left_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {left_path}")
    if show:
        plt.show()
    else:
        plt.close(fig1)

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    ax2.imshow(bg, extent=extent, origin="upper", aspect="auto")
    ax2.set_xlim(0, maze_shape[1])
    ax2.set_ylim(maze_shape[0], 0)
    ax2.set_aspect("equal")
    ax2.set_title("右臂运动轨迹 (Top / Right Arm)", fontsize=14)
    ax2.set_xlabel("Maze X")
    ax2.set_ylabel("Maze Y")

    ax2.plot(path_x, path_y, "k-", linewidth=1.5, alpha=0.5, label="Center path", zorder=1)
    ax2.plot(top_x, top_y, color="orange", linewidth=2.5, label="右臂轨迹", zorder=2)
    ax2.scatter(top_x, top_y, c="C1", s=20, alpha=0.8, zorder=3)
    for i in switch_indices:
        if waypoints[i].master == ArmRole.TOP:
            ax2.scatter(top_x[i], top_y[i], c="red", s=100, marker="x", zorder=4)
    ax2.scatter(path_x[0], path_y[0], c="green", s=150, marker="o", label="Start", zorder=5)
    ax2.scatter(top_x[-1], top_y[-1], c="red", s=150, marker="*", label="Goal", zorder=5)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    right_path = out_dir / "visualization_right_arm.png"
    fig2.savefig(right_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {right_path}")
    if show:
        plt.show()
    else:
        plt.close(fig2)

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize navigation trajectory and arm switching")
    parser.add_argument("--dataset", default="robot_data_20260220_195724", help="Dataset name")
    parser.add_argument("--target", default="A", help="Target name (A, B, C)")
    parser.add_argument("--config", default=None, help="Config JSON path")
    parser.add_argument("--output", "-o", default=None, help="Output figure path (or dir for --arms)")
    parser.add_argument("--arms", action="store_true", help="Generate left/right arm trajectory figures")
    parser.add_argument("--no-show", action="store_true", help="Do not show interactive window")
    args = parser.parse_args()

    if args.arms:
        visualize_arm_trajectories(
            dataset_name=args.dataset,
            target_name=args.target,
            config_path=args.config,
            output_dir=args.output,
            show=not args.no_show,
        )
    else:
        visualize_navigation(
            dataset_name=args.dataset,
            target_name=args.target,
            config_path=args.config,
            output_path=args.output,
            show=not args.no_show,
        )

if __name__ == "__main__":
    main()
