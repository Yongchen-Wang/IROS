from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

if not os.getenv("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import NavigationConfig, load_config
from .costmap_builder import build_safety_costmap
from .data_loader import load_navigation_data
from .path_planner import PathResult, plan_best_path_from_region
from ..paths import CONFIG_ROOT, OUTPUT_ROOT, PROJECT_ROOT

def _compute_tangents(
    path_maze: np.ndarray,
    initial_tangent: Optional[np.ndarray] = None,
) -> np.ndarray:

    n = len(path_maze)
    if n < 2:
        tangents = np.zeros((n, 2), dtype=float)
        if n == 1 and initial_tangent is not None:
            d = np.asarray(initial_tangent, dtype=float)
            norm = float(np.hypot(d[0], d[1]))
            if norm > 1e-10:
                tangents[0] = d / norm
        return tangents

    tangents = np.zeros((n, 2), dtype=float)

    norm_init = None
    if initial_tangent is not None:
        d0 = np.asarray(initial_tangent, dtype=float)
        norm_init = float(np.hypot(d0[0], d0[1]))
        if norm_init > 1e-10:
            d0 /= norm_init
        else:
            norm_init = None

    for i in range(n):
        if i == 0:
            if norm_init is not None:
                d = d0
            else:
                d = path_maze[1] - path_maze[0]
        elif i == n - 1:
            d = path_maze[-1] - path_maze[-2]
        else:
            d = path_maze[i + 1] - path_maze[i - 1]

        norm = np.hypot(d[0], d[1])
        if norm > 1e-10:
            tangents[i] = d / norm
        else:
            tangents[i] = tangents[i - 1] if i > 0 else np.array([0.0, 0.0])

    return tangents

def _compute_fixed_endpoints(
    path_maze: np.ndarray,
    robot_length_px: float,
    initial_tangent: Optional[np.ndarray] = None,
) -> tuple:

    n = len(path_maze)
    half_len = 0.5 * robot_length_px
    tangents = _compute_tangents(path_maze, initial_tangent=initial_tangent)

    bottom_maze = np.zeros((n, 2), dtype=float)
    top_maze = np.zeros((n, 2), dtype=float)

    for i in range(n):
        if i == 0:
            dx, dy = 0.0, 1.0
        else:
            dy, dx = tangents[i]

        bottom_maze[i, 0] = path_maze[i, 0] - dy * half_len
        bottom_maze[i, 1] = path_maze[i, 1] - dx * half_len
        top_maze[i, 0] = path_maze[i, 0] + dy * half_len
        top_maze[i, 1] = path_maze[i, 1] + dx * half_len

    return bottom_maze, top_maze, tangents

def _resample_path(path_maze: np.ndarray, resolution: float) -> np.ndarray:

    if len(path_maze) < 2 or resolution <= 0:
        return path_maze

    diffs = np.diff(path_maze, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cum_lengths[-1]

    if total_len < 1e-10:
        return path_maze

    n_new = max(2, int(np.ceil(total_len / resolution)) + 1)
    target_lengths = np.linspace(0, total_len, n_new, endpoint=True)

    new_path = []
    for t in target_lengths:
        idx = np.searchsorted(cum_lengths, t, side="right") - 1
        idx = min(idx, len(path_maze) - 2)
        if cum_lengths[idx + 1] - cum_lengths[idx] < 1e-10:
            alpha = 0.0
        else:
            alpha = (t - cum_lengths[idx]) / (cum_lengths[idx + 1] - cum_lengths[idx])
        alpha = float(np.clip(alpha, 0.0, 1.0))
        pt = (1 - alpha) * path_maze[idx] + alpha * path_maze[idx + 1]
        new_path.append(pt)

    return np.array(new_path)

def _smooth_path(path_maze: np.ndarray, smoothing_factor: float = 0.3) -> np.ndarray:

    from scipy.interpolate import splprep, splev

    if len(path_maze) < 3:
        return path_maze

    path_t = path_maze.T
    diffs = np.diff(path_maze, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cum_lengths[-1]

    if total_len < 1e-10:
        return path_maze

    u = cum_lengths / total_len if total_len > 0 else np.linspace(0, 1, len(path_maze))

    try:
        tck, u_smooth = splprep(
            [path_t[0], path_t[1]],
            u=u,
            s=len(path_maze) * smoothing_factor * total_len,
            k=min(3, len(path_maze) - 1),
            per=0,
        )
        u_eval = np.linspace(0, 1, len(path_maze))
        smoothed = splev(u_eval, tck)
        smoothed_path = np.column_stack([smoothed[0], smoothed[1]])
        smoothed_path[0] = path_maze[0]
        smoothed_path[-1] = path_maze[-1]
        return smoothed_path
    except Exception:
        return path_maze

def visualize_dual_arm_planner_gif(
    dataset_name: str = "robot_data_20260301_A",
    target_name: str = "A",
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
    costmap_alpha: float = 0.5,
    label_fontsize: int = 28,
    legend_fontsize: int = 28,
    title_fontsize: int = 30,
    fps: int = 8,
    total_frames: int = 40,
) -> None:

    project_root = PROJECT_ROOT
    config = load_config(config_path or str(CONFIG_ROOT / "navigation" / "default.json"))

    print(f"Loading dataset: {dataset_name}")
    data = load_navigation_data(dataset_name, config)

    print(f"Planning path to target '{target_name}'")
    path_result = plan_best_path_from_region(
        data=data,
        config=config,
        target_name=target_name,
        rng=None,
    )

    path_maze = np.asarray(path_result.path_maze, dtype=float)
    initial_tangent = np.array([1.0, 0.0])

    if config.path_resolution > 0 and len(path_maze) >= 2:
        path_maze = _resample_path(path_maze, config.path_resolution)

    if config.path_smoothing_enabled and len(path_maze) >= 3:
        path_maze = _smooth_path(path_maze, config.path_smoothing_factor)

    if data.ump_rod_length_maze is not None:
        robot_length_px = float(data.ump_rod_length_maze)
    else:
        robot_length_px = 2.0 * float(data.robot_radius_px)

    free_mask = data.maze_extractor.free_mask
    valid_indices = []
    for i in range(len(path_maze)):
        cy, cx = int(round(path_maze[i, 0])), int(round(path_maze[i, 1]))
        if 0 <= cy < free_mask.shape[0] and 0 <= cx < free_mask.shape[1]:
            if free_mask[cy, cx]:
                valid_indices.append(i)

    if valid_indices:
        last_valid = valid_indices[-1]
        if last_valid < len(path_maze) - 1:
            path_maze = path_maze[:last_valid + 1]

    config.targets_maze[target_name] = (float(path_maze[-1, 0]), float(path_maze[-1, 1]))

    maze_shape = tuple(data.maze_extractor.free_mask.shape)

    costmap = build_safety_costmap(
        maze_extractor=data.maze_extractor,
        robot_radius_px=data.robot_radius_px,
        config=config,
        robot_length_px=data.ump_rod_length_maze,
    )
    costs_display = np.ma.masked_where(~costmap.valid_mask, costmap.costs)
    costs_display = np.ma.masked_where(costs_display == np.inf, costs_display)

    free_mask = np.asarray(data.maze_extractor.free_mask)
    bg = np.where(free_mask[:, :, np.newaxis], [1.0, 1.0, 1.0], [0.85, 0.85, 0.85])
    extent = [0, maze_shape[1], maze_shape[0], 0]

    endpoint_size = 200
    line_width = 9
    center_size = 80
    half_len = robot_length_px / 2.0

    tangents = _compute_tangents(path_maze, initial_tangent=initial_tangent)

    frames = []
    n_points = len(path_maze)

    frame_indices = np.linspace(0, n_points - 1, total_frames, dtype=int)

    for frame_idx, i in enumerate(frame_indices):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        ax.imshow(bg, extent=extent, origin="upper", aspect="auto")

        im = ax.imshow(costs_display, cmap="YlGnBu", alpha=costmap_alpha,
                      extent=extent, origin="upper", aspect="auto", vmin=0, vmax=5)

        ax.set_xlim(0, maze_shape[1])
        ax.set_ylim(maze_shape[0], 0)
        ax.set_aspect("equal")
        ax.set_title(f"Dual-arm Path Planning (Target {target_name})",
                     fontsize=title_fontsize, y=1.02)
        ax.set_xlabel("X (pixels)", fontsize=label_fontsize)
        ax.set_ylabel("Y (pixels)", fontsize=label_fontsize)

        path_x = path_maze[:i+1, 1]
        path_y = path_maze[:i+1, 0]
        ax.plot(path_x, path_y, "k-", linewidth=2.0, alpha=0.7, zorder=2, label="Path")

        if i < n_points - 1:
            remaining_x = path_maze[i+1:, 1]
            remaining_y = path_maze[i+1:, 0]
            ax.plot(remaining_x, remaining_y, "k--", linewidth=1.0, alpha=0.2, zorder=2)

        center = path_maze[i]
        center_x, center_y = center[1], center[0]

        if i == 0:
            dx, dy = 0.0, 1.0
        else:
            dy, dx = tangents[i]
            mag = np.sqrt(dx * dx + dy * dy)
            if mag > 0:
                dx, dy = dx / mag, dy / mag

        bottom_x = center_x - dx * half_len
        bottom_y = center_y - dy * half_len
        top_x = center_x + dx * half_len
        top_y = center_y + dy * half_len

        ax.plot(
            [bottom_x, top_x],
            [bottom_y, top_y],
            "g-", linewidth=line_width, alpha=0.9, zorder=4
        )

        ax.scatter(bottom_x, bottom_y, c="blue", s=endpoint_size,
                   alpha=0.9, zorder=5, edgecolors="black", linewidths=1.5)

        ax.scatter(top_x, top_y, c="orange", s=endpoint_size,
                   alpha=0.9, zorder=5, edgecolors="black", linewidths=1.5)

        ax.scatter(center_x, center_y, c="black", s=center_size, alpha=0.9,
                   zorder=5, marker="o")

        start = path_maze[0]
        ax.scatter(start[1], start[0], c="hotpink", s=250, marker="s",
                   label="Start", zorder=5, edgecolors="black", linewidths=1.5)

        goal = path_maze[-1]
        ax.scatter(goal[1], goal[0], c="orange", s=400, marker="*",
                   label="Goal", zorder=5, edgecolors="black", linewidths=2.0)

        ax.legend(loc="upper center", fontsize=legend_fontsize, framealpha=0.9)
        ax.tick_params(axis='both', which='major', labelsize=label_fontsize - 2)
        ax.grid(True, alpha=0.3, linestyle='--')

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Travel Cost", fontsize=label_fontsize - 2)
        cbar.ax.tick_params(labelsize=label_fontsize - 2)

        plt.tight_layout()

        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())

        image_rgb = image[:, :, :3]
        frames.append(Image.fromarray(image_rgb))
        plt.close()

    if not output_path:
        output_dir = OUTPUT_ROOT / "navigation"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"visualization_dual_arm_{dataset_name}_{target_name}_{timestamp}.gif")

    print(f"Saving GIF with {len(frames)} frames...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"Saved: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate animated GIF of dual-arm path planning"
    )
    parser.add_argument(
        "--dataset", default="robot_data_20260301_A",
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
        help="Output GIF path"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Costmap transparency (0-1)"
    )
    parser.add_argument(
        "--fps", type=int, default=8,
        help="Frames per second for GIF"
    )
    parser.add_argument(
        "--frames", type=int, default=40,
        help="Total number of frames in GIF"
    )
    args = parser.parse_args()

    visualize_dual_arm_planner_gif(
        dataset_name=args.dataset,
        target_name=args.target,
        config_path=args.config,
        output_path=args.output,
        costmap_alpha=args.alpha,
        fps=args.fps,
        total_frames=args.frames,
    )

if __name__ == "__main__":
    main()
