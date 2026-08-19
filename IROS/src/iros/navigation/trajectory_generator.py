from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import splprep, splev

from .arm_switching import ArmRole, ArmSwitchResult, check_rod_collision, compute_arm_switching
from .config import NavigationConfig
from .coordinate_transform import cam_segment_to_maze, maze_to_cam
from .data_loader import NavigationData
from .path_planner import PathResult, plan_best_path_from_region
from .ump_geometry import (
    generate_rod_mask_for_path,
    check_path_rod_collision,
    generate_dual_rod_masks,
    check_dual_arm_trajectory_collision,
)

@dataclass
class TrajectoryWaypoint:

    center_maze: Tuple[float, float]
    bottom_maze: Tuple[float, float]
    top_maze: Tuple[float, float]
    center_cam: Tuple[float, float]
    bottom_cam: Tuple[float, float]
    top_cam: Tuple[float, float]
    master: ArmRole
    tangent_maze: Tuple[float, float]                        

@dataclass
class FullTrajectory:

    waypoints: List[TrajectoryWaypoint]
    path_result: PathResult
    arm_switch_result: ArmSwitchResult
    switch_indices: List[int]
    collision_indices: List[Tuple[int, int]]
    target_name: str
    maze_shape: Tuple[int, int]
    maze_params: Dict[str, Any]

    @property
    def n_waypoints(self) -> int:
        return len(self.waypoints)

    def to_dict(self) -> Dict[str, Any]:

        waypoints_data = []
        for wp in self.waypoints:
            waypoints_data.append({
                "center_maze": list(wp.center_maze),
                "bottom_maze": list(wp.bottom_maze),
                "top_maze": list(wp.top_maze),
                "center_cam": list(wp.center_cam),
                "bottom_cam": list(wp.bottom_cam),
                "top_cam": list(wp.top_cam),
                "master": wp.master.value,
                "tangent_maze": list(wp.tangent_maze),
            })
        return {
            "waypoints": waypoints_data,
            "path_maze": self.path_result.path_maze.tolist(),
            "switch_indices": self.switch_indices,
            "collision_indices": self.collision_indices,
            "target_name": self.target_name,
            "maze_shape": list(self.maze_shape),
            "maze_params": self.maze_params,
        }

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

def _build_waypoints(
    arm_switch_result: ArmSwitchResult,
    maze_params: Dict[str, Any],
    maze_shape: Tuple[int, int],
) -> List[TrajectoryWaypoint]:

    traj = arm_switch_result.trajectories
    path_maze = traj.path_maze
    bottom_maze = traj.bottom_maze
    top_maze = traj.top_maze
    tangents = traj.tangents
    master_seq = arm_switch_result.master_sequence

    waypoints: List[TrajectoryWaypoint] = []
    mh, mw = maze_shape

    for i in range(len(path_maze)):
        cy, cx = path_maze[i][0], path_maze[i][1]
        by, bx = bottom_maze[i][0], bottom_maze[i][1]
        ty, tx = top_maze[i][0], top_maze[i][1]
        dy, dx = tangents[i][0], tangents[i][1]

        center_cam = maze_to_cam(cy, cx, maze_params, maze_shape)
        bottom_cam = maze_to_cam(by, bx, maze_params, maze_shape)
        top_cam = maze_to_cam(ty, tx, maze_params, maze_shape)

        wp = TrajectoryWaypoint(
            center_maze=(float(cy), float(cx)),
            bottom_maze=(float(by), float(bx)),
            top_maze=(float(ty), float(tx)),
            center_cam=center_cam,
            bottom_cam=bottom_cam,
            top_cam=top_cam,
            master=master_seq[i],
            tangent_maze=(float(dy), float(dx)),
        )
        waypoints.append(wp)

    return waypoints

def _initial_pose_from_tracking(
    data: NavigationData,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

    df = data.tracking_df
    if df is None or df.empty:
        return None, None

    row0 = df.iloc[0]
    required_cols = {
        "bottom_x",
        "bottom_y",
        "top_x",
        "top_y",
        "centroid_x",
        "centroid_y",
    }
    if not required_cols.issubset(row0.index):
        return None, None

    try:
        maze_shape = tuple(data.maze_extractor.free_mask.shape)                            
    except Exception:
        return None, None

    try:
        seg_mazecoords = cam_segment_to_maze(
            bottom_x=float(row0["bottom_x"]),
            bottom_y=float(row0["bottom_y"]),
            top_x=float(row0["top_x"]),
            top_y=float(row0["top_y"]),
            centroid_x=float(row0["centroid_x"]),
            centroid_y=float(row0["centroid_y"]),
            maze_params=data.maze_params,
            maze_shape=maze_shape,
        )
    except Exception:
        return None, None

    bottom_maze = np.asarray(seg_mazecoords["bottom"], dtype=float)
    top_maze = np.asarray(seg_mazecoords["top"], dtype=float)
    center_maze = np.asarray(seg_mazecoords["centroid"], dtype=float)

    d = top_maze - bottom_maze
    norm = float(np.hypot(d[0], d[1]))
    if norm <= 1e-6:
        tangent_maze = None
    else:
        tangent_maze = d / norm

    return center_maze, tangent_maze

def generate_full_trajectory(
    data: NavigationData,
    config: NavigationConfig,
    target_name: str,
    path_result: Optional[PathResult] = None,
    rng: Optional[np.random.Generator] = None,
    initial_master: Optional[ArmRole] = None,
) -> FullTrajectory:

    if path_result is None:
        path_result = plan_best_path_from_region(
            data=data,
            config=config,
            target_name=target_name,
            rng=rng,
        )

    path_maze = np.asarray(path_result.path_maze, dtype=float)

    _, init_tangent_maze = _initial_pose_from_tracking(data)

    if config.path_resolution > 0 and len(path_maze) >= 2:
        path_maze = _resample_path(path_maze, config.path_resolution)
        path_result = PathResult(
            start_maze=path_result.start_maze,
            goal_maze=path_result.goal_maze,
            path_maze=path_maze,
            total_cost=path_result.total_cost,
        )

    if config.path_smoothing_enabled and len(path_maze) >= 3:
        path_maze = _smooth_path(path_maze, config.path_smoothing_factor)
        path_result = PathResult(
            start_maze=path_result.start_maze,
            goal_maze=path_result.goal_maze,
            path_maze=path_maze,
            total_cost=path_result.total_cost,
        )

    if data.ump_rod_length_maze is not None:
        robot_length_px = float(data.ump_rod_length_maze)
    else:
        robot_length_px = 2.0 * float(data.robot_radius_px)

    max_switches = getattr(config, "max_arm_switches", 1)
    arm_switch_result = compute_arm_switching(
        path_result=path_result,
        robot_length_px=robot_length_px,
        switch_threshold=20.0,                                                          
        max_switches=max_switches,                                           
        initial_master=initial_master,

        initial_tangent=init_tangent_maze,
    )

    collision_indices = check_rod_collision(arm_switch_result.trajectories)

    maze_shape = tuple(data.maze_extractor.free_mask.shape)                            
    maze_params = data.maze_params

    ump_rod_collisions = []
    if config.enable_ump_rod_detection and data.ump_rod_width_maze is not None and data.ump_rod_length_maze is not None:
        free_mask = np.asarray(data.maze_extractor.free_mask, dtype=bool)

        trajectories = arm_switch_result.trajectories
        bottom_endpoints = trajectories.bottom_maze
        top_endpoints = trajectories.top_maze

        left_rod_mask, right_rod_mask = generate_dual_rod_masks(
            bottom_endpoints=bottom_endpoints,
            top_endpoints=top_endpoints,
            rod_width_maze=data.ump_rod_width_maze,
            maze_shape=maze_shape,
        )

        has_dual_collision, collision_steps, overlap_mask = check_dual_arm_trajectory_collision(
            bottom_endpoints=bottom_endpoints,
            top_endpoints=top_endpoints,
            rod_width_maze=data.ump_rod_width_maze,
            maze_shape=maze_shape,
        )

        if has_dual_collision:
            overlap_pixels = int(np.sum(overlap_mask))
            preview = ", ".join(str(i) for i in collision_steps[:10])
            if len(collision_steps) > 10:
                preview += ", ..."
            raise RuntimeError(
                "Dual-arm collision detected after trajectory generation; refusing "
                "to return an unsafe trajectory. "
                f"Synchronous collision waypoint(s): [{preview}] "
                f"({len(collision_steps)} total, {overlap_pixels} overlapping pixels)."
            )

        path_rod_mask = left_rod_mask | right_rod_mask

        ump_rod_collisions = check_path_rod_collision(
            path_maze=path_result.path_maze,
            rod_mask=path_rod_mask,
            free_mask=free_mask,
        )

        if ump_rod_collisions:
            import warnings
            warnings.warn(
                f"UMP rod collision with walls detected at {len(ump_rod_collisions)} path point(s) in trajectory. "
                f"Trajectory may not be safe for execution."
            )

    waypoints = _build_waypoints(
        arm_switch_result=arm_switch_result,
        maze_params=maze_params,
        maze_shape=maze_shape,
    )

    return FullTrajectory(
        waypoints=waypoints,
        path_result=path_result,
        arm_switch_result=arm_switch_result,
        switch_indices=arm_switch_result.switch_indices,
        collision_indices=collision_indices,
        target_name=target_name,
        maze_shape=maze_shape,
        maze_params=maze_params,
    )
