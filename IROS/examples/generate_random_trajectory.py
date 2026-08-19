#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from iros.navigation.arm_switching import ArmRole
from iros.navigation.config import CircularStartRegion, NavigationConfig, load_config
from iros.navigation.coordinate_transform import cam_to_maze, maze_to_cam
from iros.navigation.costmap_builder import build_safety_costmap
from iros.navigation.data_loader import load_navigation_data
from iros.navigation.path_planner import PathResult, _astar_on_grid, _nearest_valid_cell
from iros.navigation.trajectory_generator import generate_full_trajectory
from iros.paths import CONFIG_ROOT, OUTPUT_ROOT, PROJECT_ROOT

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

def sample_random_start_point(
    data,
    config: NavigationConfig,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:

    if rng is None:
        rng = np.random.default_rng()

    costmap = build_safety_costmap(
        maze_extractor=data.maze_extractor,
        robot_radius_px=data.robot_radius_px,
        config=config,
        robot_length_px=data.ump_rod_length_maze,
    )

    maze_shape = costmap.maze_shape
    maze_params = data.maze_params

    if config.start_region is not None:
        from iros.navigation.path_planner import _sample_start_points_in_camera

        cam_samples = _sample_start_points_in_camera(
            start_region=config.start_region,
            n_samples=1,
            rng=rng,
        )
        cam_x, cam_y = cam_samples[0]
        maze_y, maze_x = cam_to_maze(
            cam_x=float(cam_x),
            cam_y=float(cam_y),
            maze_params=maze_params,
            maze_shape=maze_shape,
        )
        start_idx = _nearest_valid_cell(costmap, maze_y, maze_x)
        if start_idx is not None:
            return (float(start_idx[0]), float(start_idx[1]))

    free_mask = np.asarray(data.maze_extractor.free_mask)
    valid_indices = np.argwhere(free_mask)

    if len(valid_indices) == 0:

        start_y, start_x = data.maze_extractor.start_point                            
        return (float(start_y), float(start_x))

    idx = rng.integers(0, len(valid_indices))
    y, x = valid_indices[idx]
    return (float(y), float(x))

def plan_path_from_random_start(
    data,
    config: NavigationConfig,
    target_name: str,
    rng: Optional[np.random.Generator] = None,
) -> PathResult:

    if rng is None:
        rng = np.random.default_rng()

    start_y, start_x = sample_random_start_point(data, config, rng)
    print(f"随机选择的起点: ({start_y:.1f}, {start_x:.1f})")

    costmap = build_safety_costmap(
        maze_extractor=data.maze_extractor,
        robot_radius_px=data.robot_radius_px,
        config=config,
        robot_length_px=data.ump_rod_length_maze,
    )

    targets: dict[str, tuple[float, float]] = data.maze_extractor.targets                            
    if target_name not in targets:
        raise KeyError(f"Target '{target_name}' not found in maze_extractor.targets.")
    goal_y, goal_x = targets[target_name]

    start_idx = _nearest_valid_cell(costmap, start_y, start_x)
    goal_idx = _nearest_valid_cell(costmap, goal_y, goal_x)

    if start_idx is None:
        raise RuntimeError("No valid start cell found.")
    if goal_idx is None:
        raise RuntimeError("No valid goal cell found near target.")

    path_result = _astar_on_grid(
        costmap=costmap,
        start_idx=start_idx,
        goal_idx=goal_idx,
    )

    if path_result is None:
        raise RuntimeError("Failed to find a path from start to goal.")

    return path_result

def visualize_three_arm_figures(
    traj,
    data,
    output_dir: Path,
    show: bool = False,
) -> None:

    project_root = PROJECT_ROOT
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

    output_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.imshow(bg, extent=extent, origin="upper", aspect="auto")
    ax1.set_xlim(0, maze_shape[1])
    ax1.set_ylim(maze_shape[0], 0)
    ax1.set_aspect("equal")
    ax1.set_title("左臂运动轨迹 (Left Arm / Bottom Endpoint)", fontsize=16, fontweight="bold")
    ax1.set_xlabel("Maze X", fontsize=12)
    ax1.set_ylabel("Maze Y", fontsize=12)

    ax1.plot(path_x, path_y, "k-", linewidth=1.5, alpha=0.4, label="中心路径", zorder=1)
    ax1.plot(bottom_x, bottom_y, "b-", linewidth=3, label="左臂轨迹", zorder=2)
    ax1.scatter(bottom_x, bottom_y, c="blue", s=30, alpha=0.7, zorder=3)
    for i in switch_indices:
        if waypoints[i].master == ArmRole.BOTTOM:
            ax1.scatter(bottom_x[i], bottom_y[i], c="red", s=150, marker="x", linewidths=3, zorder=4, label="切换点" if i == min(switch_indices) else "")
    ax1.scatter(path_x[0], path_y[0], c="green", s=200, marker="o", label="起点", zorder=5, edgecolors="darkgreen", linewidths=2)
    ax1.scatter(bottom_x[-1], bottom_y[-1], c="red", s=200, marker="*", label="终点", zorder=5, edgecolors="darkred", linewidths=2)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    left_path = output_dir / "left_arm_trajectory.png"
    fig1.savefig(left_path, dpi=150, bbox_inches="tight")
    print(f"✓ 已保存: {left_path}")
    if show:
        plt.show()
    else:
        plt.close(fig1)

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.imshow(bg, extent=extent, origin="upper", aspect="auto")
    ax2.set_xlim(0, maze_shape[1])
    ax2.set_ylim(maze_shape[0], 0)
    ax2.set_aspect("equal")
    ax2.set_title("右臂运动轨迹 (Right Arm / Top Endpoint)", fontsize=16, fontweight="bold")
    ax2.set_xlabel("Maze X", fontsize=12)
    ax2.set_ylabel("Maze Y", fontsize=12)

    ax2.plot(path_x, path_y, "k-", linewidth=1.5, alpha=0.4, label="中心路径", zorder=1)
    ax2.plot(top_x, top_y, color="orange", linewidth=3, label="右臂轨迹", zorder=2)
    ax2.scatter(top_x, top_y, c="orange", s=30, alpha=0.7, zorder=3)
    for i in switch_indices:
        if waypoints[i].master == ArmRole.TOP:
            ax2.scatter(top_x[i], top_y[i], c="red", s=150, marker="x", linewidths=3, zorder=4, label="切换点" if i == min(switch_indices) else "")
    ax2.scatter(path_x[0], path_y[0], c="green", s=200, marker="o", label="起点", zorder=5, edgecolors="darkgreen", linewidths=2)
    ax2.scatter(top_x[-1], top_y[-1], c="red", s=200, marker="*", label="终点", zorder=5, edgecolors="darkred", linewidths=2)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    right_path = output_dir / "right_arm_trajectory.png"
    fig2.savefig(right_path, dpi=150, bbox_inches="tight")
    print(f"✓ 已保存: {right_path}")
    if show:
        plt.show()
    else:
        plt.close(fig2)

    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
    ax3.imshow(bg, extent=extent, origin="upper", aspect="auto")
    ax3.set_xlim(0, maze_shape[1])
    ax3.set_ylim(maze_shape[0], 0)
    ax3.set_aspect("equal")
    ax3.set_title("双臂运动轨迹 (Both Arms)", fontsize=16, fontweight="bold")
    ax3.set_xlabel("Maze X", fontsize=12)
    ax3.set_ylabel("Maze Y", fontsize=12)

    ax3.plot(path_x, path_y, "k-", linewidth=2, alpha=0.5, label="中心路径", zorder=1)
    ax3.plot(bottom_x, bottom_y, "b-", linewidth=2.5, label="左臂轨迹", zorder=2)
    ax3.plot(top_x, top_y, color="orange", linewidth=2.5, label="右臂轨迹", zorder=2)
    ax3.scatter(bottom_x, bottom_y, c="blue", s=20, alpha=0.6, zorder=3)
    ax3.scatter(top_x, top_y, c="orange", s=20, alpha=0.6, zorder=3)

    for i in switch_indices:
        ax3.scatter(path_x[i], path_y[i], c="red", s=150, marker="x", linewidths=3, zorder=4)

    ax3.scatter(path_x[0], path_y[0], c="green", s=200, marker="o", label="起点", zorder=5, edgecolors="darkgreen", linewidths=2)
    ax3.scatter(path_x[-1], path_y[-1], c="red", s=200, marker="*", label="终点", zorder=5, edgecolors="darkred", linewidths=2)
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    both_path = output_dir / "both_arms_trajectory.png"
    fig3.savefig(both_path, dpi=150, bbox_inches="tight")
    print(f"✓ 已保存: {both_path}")
    if show:
        plt.show()
    else:
        plt.close(fig3)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="随机选择起点，规划到 targetA 的路径，生成三张可视化图"
    )
    parser.add_argument(
        "--dataset",
        default="robot_data_20260220_195724",
        help="数据集名称",
    )
    parser.add_argument(
        "--target",
        default="A",
        help="目标名称 (A, B, C)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（JSON）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="输出目录（默认: outputs/navigation）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（用于可重复性）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="显示图像窗口",
    )
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    config_path = args.config or str(CONFIG_ROOT / "navigation" / "example.json")
    config = load_config(config_path)

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / "navigation"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        print(f"使用随机种子: {args.seed}")

    print(f"加载数据集: {args.dataset}")
    data = load_navigation_data(args.dataset, config)

    print(f"规划到目标 '{args.target}' 的路径...")
    path_result = plan_path_from_random_start(
        data=data,
        config=config,
        target_name=args.target,
        rng=rng,
    )

    print(f"生成完整轨迹...")
    traj = generate_full_trajectory(
        data=data,
        config=config,
        target_name=args.target,
        path_result=path_result,
        rng=rng,
    )

    print(f"生成可视化图...")
    visualize_three_arm_figures(
        traj=traj,
        data=data,
        output_dir=output_dir,
        show=args.show,
    )

    print(f"\n完成！所有图像已保存到: {output_dir}")

if __name__ == "__main__":
    main()
