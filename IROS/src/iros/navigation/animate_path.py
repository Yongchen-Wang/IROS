from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .arm_switching import ArmRole
from .config import NavigationConfig, load_config
from .data_loader import load_navigation_data
from .trajectory_generator import generate_full_trajectory
from .ump_geometry import generate_rod_mask_for_path, generate_dual_rod_masks
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
        img = np.asarray(Image.open(p))
        if img.ndim == 3:
            return img
        return np.stack([img] * 3, axis=-1)
    except Exception:
        return None

def create_path_animation(
    dataset_name: str = "robot_data_20260220_195724",
    target_name: str = "A",
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
    fps: int = 10,
    frame_skip: int = 1,
    initial_master: Optional[ArmRole] = None,
) -> None:

    project_root = PROJECT_ROOT

    config = load_config(config_path or str(CONFIG_ROOT / "navigation" / "default.json"))

    print(f"Loading dataset: {dataset_name}")
    data = load_navigation_data(dataset_name, config)

    print(f"Generating trajectory to target '{target_name}'")
    if initial_master is not None:
        print(f"  Using initial master: {initial_master.value} arm")
    traj = generate_full_trajectory(
        data, 
        config, 
        target_name=target_name,
        initial_master=initial_master,
    )

    maze_shape = traj.maze_shape
    maze_params = traj.maze_params
    waypoints = traj.waypoints
    switch_indices = set(traj.switch_indices)

    ump_rod_enabled = False
    ump_rod_width_maze = None
    bottom_endpoints_all = None
    top_endpoints_all = None

    if config.enable_ump_rod_detection:

        bottom_endpoints_all = np.array([[wp.bottom_maze[0], wp.bottom_maze[1]] for wp in waypoints])
        top_endpoints_all = np.array([[wp.top_maze[0], wp.top_maze[1]] for wp in waypoints])

        if hasattr(data, 'ump_rod_width_maze') and data.ump_rod_width_maze is not None:
            ump_rod_width_maze = data.ump_rod_width_maze
            print("Preparing UMP rod mask data for dynamic visualization...")
            print(f"UMP rod visualization enabled: width={ump_rod_width_maze:.2f} pixels (detected)")
        else:

            ump_rod_width_maze = 10.0                                
            print("Preparing UMP rod mask data for dynamic visualization...")
            print(f"UMP rod visualization enabled: width={ump_rod_width_maze:.2f} pixels (default, detection failed)")

        ump_rod_enabled = True

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

    indices = list(range(0, len(waypoints), frame_skip))
    if indices[-1] != len(waypoints) - 1:
        indices.append(len(waypoints) - 1)                                

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(bg, extent=extent, origin="upper", aspect="auto")
    ax.set_xlim(0, maze_shape[1])
    ax.set_ylim(maze_shape[0], 0)
    ax.set_aspect("equal")

    master_label = ""
    if initial_master is not None:
        master_label = f" ({initial_master.value} arm leading)"
    ax.set_title(
        f"Navigation Animation: {dataset_name} → Target {target_name}{master_label}",
        fontsize=14
    )
    ax.set_xlabel("Maze X")
    ax.set_ylabel("Maze Y")

    left_rod_overlay = None
    right_rod_overlay = None
    if ump_rod_enabled:

        left_rod_overlay = ax.imshow(
            np.zeros((maze_shape[0], maze_shape[1], 4), dtype=np.float32),
            extent=extent, origin="upper", aspect="auto", zorder=2, alpha=0.4
        )
        right_rod_overlay = ax.imshow(
            np.zeros((maze_shape[0], maze_shape[1], 4), dtype=np.float32),
            extent=extent, origin="upper", aspect="auto", zorder=2, alpha=0.4
        )

    ax.plot(path_x, path_y, "k-", linewidth=1.5, alpha=0.3, label="Planned path", zorder=1)
    ax.plot(bottom_x, bottom_y, "b--", linewidth=1, alpha=0.2, label="Bottom endpoint path", zorder=1)
    ax.plot(top_x, top_y, "orange", linestyle="--", linewidth=1, alpha=0.2, label="Top endpoint path", zorder=1)

    start_marker = ax.scatter([], [], c="green", s=150, marker="o", label="Start", zorder=10)
    goal_marker = ax.scatter([], [], c="red", s=150, marker="*", label="Goal", zorder=10)

    center_point = ax.scatter([], [], c="blue", s=80, marker="o", zorder=8, label="Robot center")

    bottom_point = ax.scatter([], [], c="cyan", s=60, marker="o", zorder=7, label="Bottom endpoint")
    top_point = ax.scatter([], [], c="orange", s=60, marker="o", zorder=7, label="Top endpoint")

    robot_body = ax.plot([], [], "g-", linewidth=3, alpha=0.7, zorder=6, label="Robot body")[0]

    trail_length = min(10, len(indices) // 2)
    trail = ax.plot([], [], "b-", linewidth=2, alpha=0.5, zorder=5)[0]

    switch_marker = ax.scatter([], [], c="red", s=120, marker="x", linewidths=3, zorder=9, label="Arm switch")

    if ump_rod_enabled:
        ax.scatter([], [], c="red", s=100, alpha=0.4, marker="s", label="Left UMP rod")
        ax.scatter([], [], c="blue", s=100, alpha=0.4, marker="s", label="Right UMP rod")

    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    trail_x = []
    trail_y = []

    def animate(frame_idx):

        if frame_idx >= len(indices):
            return

        i = indices[frame_idx]
        wp = waypoints[i]

        center_point.set_offsets([[wp.center_maze[1], wp.center_maze[0]]])

        bottom_point.set_offsets([[wp.bottom_maze[1], wp.bottom_maze[0]]])
        top_point.set_offsets([[wp.top_maze[1], wp.top_maze[0]]])

        if wp.master == ArmRole.TOP:
            top_point.set_color("yellow")
            top_point.set_sizes([90])
            bottom_point.set_color("cyan")
            bottom_point.set_sizes([50])

            robot_body.set_color("yellowgreen")
        else:
            bottom_point.set_color("yellow")
            bottom_point.set_sizes([90])
            top_point.set_color("orange")
            top_point.set_sizes([50])

            robot_body.set_color("deepskyblue")

        robot_body.set_data(
            [wp.bottom_maze[1], wp.top_maze[1]],
            [wp.bottom_maze[0], wp.top_maze[0]]
        )

        trail_x.append(wp.center_maze[1])
        trail_y.append(wp.center_maze[0])
        if len(trail_x) > trail_length:
            trail_x.pop(0)
            trail_y.pop(0)
        trail.set_data(trail_x, trail_y)

        if frame_idx == 0:
            start_marker.set_offsets([[path_x[0], path_y[0]]])
        else:
            start_marker.set_offsets(np.empty((0, 2)))

        if frame_idx == len(indices) - 1:
            goal_marker.set_offsets([[path_x[-1], path_y[-1]]])
        else:
            goal_marker.set_offsets(np.empty((0, 2)))

        if i in switch_indices:
            switch_marker.set_offsets([[wp.center_maze[1], wp.center_maze[0]]])
        else:
            switch_marker.set_offsets(np.empty((0, 2)))

        if ump_rod_enabled and left_rod_overlay is not None and right_rod_overlay is not None:

            bottom_endpoints_window = bottom_endpoints_all[i : i + 1]
            top_endpoints_window = top_endpoints_all[i : i + 1]

            left_mask, right_mask = generate_dual_rod_masks(
                bottom_endpoints=bottom_endpoints_window,
                top_endpoints=top_endpoints_window,
                rod_width_maze=ump_rod_width_maze,
                maze_shape=maze_shape,
            )

            if initial_master == ArmRole.TOP:

                left_alpha = np.where(left_mask & ~right_mask, 0.4, 0.0)
                right_alpha = np.where(right_mask, 0.4, 0.0)
            elif initial_master == ArmRole.BOTTOM:

                left_alpha = np.where(left_mask, 0.4, 0.0)
                right_alpha = np.where(right_mask & ~left_mask, 0.4, 0.0)
            else:

                left_alpha = np.where(left_mask, 0.4, 0.0)
                right_alpha = np.where(right_mask, 0.4, 0.0)

            left_overlay_img = np.zeros((maze_shape[0], maze_shape[1], 4), dtype=np.float32)
            left_overlay_img[..., 0] = 1.0       
            left_overlay_img[..., 3] = left_alpha

            right_overlay_img = np.zeros((maze_shape[0], maze_shape[1], 4), dtype=np.float32)
            right_overlay_img[..., 2] = 1.0        
            right_overlay_img[..., 3] = right_alpha

            left_rod_overlay.set_array(left_overlay_img)
            right_rod_overlay.set_array(right_overlay_img)

        progress = (frame_idx + 1) / len(indices) * 100
        master_label = ""
        if initial_master is not None:
            master_label = f" ({initial_master.value} arm leading)"
        ax.set_title(
            f"Navigation Animation: {dataset_name} → Target {target_name}{master_label} "
            f"({progress:.1f}% complete)",
            fontsize=14
        )

        if ump_rod_enabled and left_rod_overlay is not None and right_rod_overlay is not None:
            return (
                center_point, bottom_point, top_point, robot_body, trail,
                start_marker, goal_marker, switch_marker,
                left_rod_overlay, right_rod_overlay
            )
        else:
            return (
                center_point, bottom_point, top_point, robot_body, trail,
                start_marker, goal_marker, switch_marker
            )

    n_frames = len(indices)
    interval_ms = 1000 / fps                          

    print(f"Creating animation with {n_frames} frames at {fps} fps...")
    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=interval_ms,
        blit=True, repeat=True
    )

    if not output_path:
        output_dir = OUTPUT_ROOT / "navigation"
        output_dir.mkdir(parents=True, exist_ok=True)
        master_suffix = ""
        if initial_master is not None:
            master_suffix = f"_{initial_master.value}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"animation_{dataset_name}_{target_name}{master_suffix}_{timestamp}.gif")

    print(f"Saving animation to: {output_path}")
    try:
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"✓ Animation saved successfully!")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Trying alternative method...")

        try:
            output_path_mp4 = output_path.replace('.gif', '.mp4')
            anim.save(output_path_mp4, writer='ffmpeg', fps=fps)
            print(f"✓ Animation saved as MP4: {output_path_mp4}")
        except Exception as e2:
            print(f"Error saving MP4: {e2}")
            print("Please install pillow or ffmpeg to save animations.")

    plt.close(fig)

def create_dual_arm_animations(
    dataset_name: str = "robot_data_20260220_195724",
    target_name: str = "A",
    config_path: Optional[str] = None,
    fps: int = 10,
    frame_skip: int = 1,
) -> None:

    print("=" * 60)
    print(f"Generating dual-arm animations for {dataset_name} → Target {target_name}")
    print("=" * 60)

    print("\n[1/2] Generating animation with RIGHT arm (TOP) leading...")
    create_path_animation(
        dataset_name=dataset_name,
        target_name=target_name,
        config_path=config_path,
        output_path=None,                           
        fps=fps,
        frame_skip=frame_skip,
        initial_master=ArmRole.TOP,
    )

    print("\n[2/2] Generating animation with LEFT arm (BOTTOM) leading...")
    create_path_animation(
        dataset_name=dataset_name,
        target_name=target_name,
        config_path=config_path,
        output_path=None,                           
        fps=fps,
        frame_skip=frame_skip,
        initial_master=ArmRole.BOTTOM,
    )

    print("\n" + "=" * 60)
    print("✓ Both animations generated successfully!")
    print("=" * 60)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate animation of robot navigation path")
    parser.add_argument("--dataset", default="robot_data_20260220_195724", help="Dataset name")
    parser.add_argument("--target", default="A", help="Target name (A, B, C)")
    parser.add_argument("--config", default=None, help="Config JSON path")
    parser.add_argument("--output", "-o", default=None, help="Output animation path")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Skip every N waypoints (default: 1 = show all)")
    parser.add_argument("--dual-arm", action="store_true", help="Generate both right and left arm leading animations")
    parser.add_argument("--initial-master", choices=["top", "bottom"], default=None, 
                       help="Force initial master arm (top=right, bottom=left). Only used if --dual-arm is not set.")
    args = parser.parse_args()

    if args.dual_arm:

        create_dual_arm_animations(
            dataset_name=args.dataset,
            target_name=args.target,
            config_path=args.config,
            fps=args.fps,
            frame_skip=args.frame_skip,
        )
    else:

        initial_master = None
        if args.initial_master == "top":
            initial_master = ArmRole.TOP
        elif args.initial_master == "bottom":
            initial_master = ArmRole.BOTTOM

        create_path_animation(
            dataset_name=args.dataset,
            target_name=args.target,
            config_path=args.config,
            output_path=args.output,
            fps=args.fps,
            frame_skip=args.frame_skip,
            initial_master=initial_master,
        )

if __name__ == "__main__":
    main()
