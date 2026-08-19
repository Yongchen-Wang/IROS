from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    CircularStartRegion,
    NavigationConfig,
    RectStartRegion,
    StartRegion,
)
from .coordinate_transform import cam_to_maze, maze_to_cam
from .costmap_builder import Costmap, build_safety_costmap
from .data_loader import NavigationData
from .ump_geometry import (
    generate_rod_mask_for_path,
    check_path_rod_collision,
    generate_dual_rod_masks,
    check_dual_arm_trajectory_collision,
)

MazePoint = Tuple[float, float]                         

@dataclass
class PathResult:

    start_maze: MazePoint
    goal_maze: MazePoint
    path_maze: np.ndarray
    total_cost: float

def _nearest_valid_cell(costmap: Costmap, y: float, x: float, max_radius: int = 50) -> Optional[Tuple[int, int]]:

    h, w = costmap.maze_shape
    iy = int(round(y))
    ix = int(round(x))
    iy = max(0, min(iy, h - 1))
    ix = max(0, min(ix, w - 1))

    if costmap.is_valid(iy, ix):
        return iy, ix

    best_cell: Optional[Tuple[int, int]] = None
    best_dist2 = float("inf")

    for r in range(1, max_radius + 1):
        y_min = max(0, iy - r)
        y_max = min(h - 1, iy + r)
        x_min = max(0, ix - r)
        x_max = min(w - 1, ix + r)

        for cy in range(y_min, y_max + 1):
            for cx in range(x_min, x_max + 1):
                if not costmap.valid_mask[cy, cx]:
                    continue
                dy = cy - y
                dx = cx - x
                dist2 = dy * dy + dx * dx
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_cell = (cy, cx)

        if best_cell is not None:
            return best_cell

    return None

def _find_nearest_valid_cell_anywhere(costmap: Costmap, y: float, x: float) -> Optional[Tuple[int, int]]:

    h, w = costmap.maze_shape
    best_cell: Optional[Tuple[int, int]] = None
    best_dist2 = float("inf")

    for cy in range(h):
        for cx in range(w):
            if not costmap.valid_mask[cy, cx]:
                continue
            dy = cy - y
            dx = cx - x
            dist2 = dy * dy + dx * dx
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_cell = (cy, cx)

    return best_cell

def _astar_on_grid(
    costmap: Costmap,
    start_idx: Tuple[int, int],
    goal_idx: Tuple[int, int],
) -> Optional[PathResult]:

    h, w = costmap.maze_shape
    sy, sx = start_idx
    gy, gx = goal_idx

    if not (costmap.is_valid(sy, sx) and costmap.is_valid(gy, gx)):
        return None

    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)),
        (-1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)),
        (1, 1, np.sqrt(2.0)),
    ]

    def heuristic(y: int, x: int) -> float:
        return float(np.hypot(gx - x, gy - y))

    open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(sy, sx), 0.0, (sy, sx)))

    g_scores = np.full((h, w), np.inf, dtype=float)
    g_scores[sy, sx] = 0.0

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    closed = np.zeros((h, w), dtype=bool)

    while open_heap:
        f, g, (y, x) = heapq.heappop(open_heap)

        if closed[y, x]:
            continue
        closed[y, x] = True

        if (y, x) == (gy, gx):

            path: List[Tuple[int, int]] = [(y, x)]
            while (y, x) in came_from:
                y, x = came_from[(y, x)]
                path.append((y, x))
            path.reverse()
            path_arr = np.array(path, dtype=float)
            return PathResult(
                start_maze=(float(sy), float(sx)),
                goal_maze=(float(gy), float(gx)),
                path_maze=path_arr,
                total_cost=float(g_scores[gy, gx]),
            )

        for dy, dx, step_len in neighbors:
            ny = y + dy
            nx = x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if closed[ny, nx]:
                continue
            if not costmap.valid_mask[ny, nx]:
                continue

            step_cost = step_len * costmap.costs[ny, nx]
            tentative_g = g + float(step_cost)

            if tentative_g < g_scores[ny, nx]:
                g_scores[ny, nx] = tentative_g
                came_from[(ny, nx)] = (y, x)
                f_score = tentative_g + heuristic(ny, nx)
                heapq.heappush(open_heap, (f_score, tentative_g, (ny, nx)))

    return None

def _sample_start_points_in_camera(
    start_region: StartRegion,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    samples = []
    if isinstance(start_region, CircularStartRegion):
        cx, cy = start_region.center
        r = float(start_region.radius)
        for _ in range(n_samples):

            u = rng.random()
            theta = 2.0 * np.pi * rng.random()
            rad = r * np.sqrt(u)
            x = cx + rad * np.cos(theta)
            y = cy + rad * np.sin(theta)
            samples.append((x, y))
    elif isinstance(start_region, RectStartRegion):
        cx, cy = start_region.center
        half_w = float(start_region.width) / 2.0
        half_h = float(start_region.height) / 2.0
        for _ in range(n_samples):
            x = rng.uniform(cx - half_w, cx + half_w)
            y = rng.uniform(cy - half_h, cy + half_h)
            samples.append((x, y))
    else:                                
        raise TypeError(f"Unsupported start_region type: {type(start_region)}")

    return np.asarray(samples, dtype=float)

def plan_best_path_from_region(
    data: NavigationData,
    config: NavigationConfig,
    target_name: str,
    rng: Optional[np.random.Generator] = None,
) -> PathResult:

    if rng is None:
        rng = np.random.default_rng()

    if config.start_region is None:
        raise ValueError(
            "NavigationConfig.start_region must be set for Navigation V1; "
            "automatic fallback to maze_extractor.start_point is disabled."
        )

    targets: Dict[str, Tuple[float, float]] = getattr(config, "targets_maze", {})
    if target_name not in targets:
        raise KeyError(
            f"Target '{target_name}' not found in config.targets_maze. "
            f"Please define it in NavigationConfig for Navigation V1."
        )
    goal_y, goal_x = targets[target_name]

    ump_rod_mask: Optional[np.ndarray] = None
    if config.enable_ump_rod_detection and data.ump_rod_width_maze is not None and data.ump_rod_length_maze is not None:

        free_mask = np.asarray(data.maze_extractor.free_mask, dtype=bool)
        maze_shape = free_mask.shape

        if config.start_region is not None:

            if isinstance(config.start_region, CircularStartRegion):
                start_cam_x, start_cam_y = config.start_region.center
            else:                   
                start_cam_x, start_cam_y = config.start_region.center
            start_y, start_x = cam_to_maze(
                cam_x=float(start_cam_x),
                cam_y=float(start_cam_y),
                maze_params=data.maze_params,
                maze_shape=maze_shape,
            )
        else:

            start_y, start_x = data.maze_extractor.start_point                            

        n_samples = 20
        path_estimate = np.zeros((n_samples, 2), dtype=float)
        for i in range(n_samples):
            alpha = i / (n_samples - 1) if n_samples > 1 else 0.0
            path_estimate[i, 0] = start_y + alpha * (goal_y - start_y)
            path_estimate[i, 1] = start_x + alpha * (goal_x - start_x)

        if data.ump_rod_length_maze is not None:
            robot_length_px = float(data.ump_rod_length_maze)
        else:
            robot_length_px = 2.0 * float(data.robot_radius_px)
        half_len = 0.5 * robot_length_px

        n = len(path_estimate)
        tangents = np.zeros((n, 2), dtype=float)
        for i in range(n):
            if i == 0:
                if n > 1:
                    d = path_estimate[1] - path_estimate[0]
                else:
                    d = np.array([0.0, 1.0])
            elif i == n - 1:
                d = path_estimate[-1] - path_estimate[-2]
            else:
                d = path_estimate[i + 1] - path_estimate[i - 1]

            norm = np.hypot(d[0], d[1])
            if norm > 1e-10:
                tangents[i] = d / norm
            else:
                tangents[i] = tangents[i - 1] if i > 0 else np.array([0.0, 1.0])

        bottom_endpoints = path_estimate - half_len * tangents
        top_endpoints = path_estimate + half_len * tangents

        left_rod_mask, right_rod_mask = generate_dual_rod_masks(
            bottom_endpoints=bottom_endpoints,
            top_endpoints=top_endpoints,
            rod_width_maze=data.ump_rod_width_maze,
            maze_shape=maze_shape,
        )

        has_collision, collision_steps, overlap_mask = check_dual_arm_trajectory_collision(
            bottom_endpoints=bottom_endpoints,
            top_endpoints=top_endpoints,
            rod_width_maze=data.ump_rod_width_maze,
            maze_shape=maze_shape,
        )

        if has_collision:
            import warnings
            overlap_pixels = int(np.sum(overlap_mask))
            warnings.warn(
                f"Dual-arm collision detected in initial path estimate at "
                f"{len(collision_steps)} synchronous waypoint(s): {overlap_pixels} overlapping pixels. "
                f"Collision regions will be marked as invalid in costmap."
            )

        ump_rod_mask = left_rod_mask | right_rod_mask

    apply_ump_mask = not bool(
        getattr(config, "disable_ump_mask_for_planning", False)
    )
    if apply_ump_mask:
        ump_rod_mask_for_costmap = ump_rod_mask
    else:
        ump_rod_mask_for_costmap = None

    costmap = build_safety_costmap(
        maze_extractor=data.maze_extractor,
        robot_radius_px=data.robot_radius_px,
        config=config,
        ump_rod_mask=ump_rod_mask_for_costmap,
    )

    goal_idx = _nearest_valid_cell(costmap, goal_y, goal_x, max_radius=200)
    if goal_idx is None:

        goal_idx = _find_nearest_valid_cell_anywhere(costmap, goal_y, goal_x)
        if goal_idx is None:

            maze_h, maze_w = costmap.maze_shape
            valid_count = int(costmap.valid_mask.sum())
            total_cells = maze_h * maze_w

            free_mask = np.asarray(data.maze_extractor.free_mask, dtype=bool)
            dt = np.asarray(data.maze_extractor.dt, dtype=float)
            free_count = int(free_mask.sum())
            if free_count > 0:
                dt_in_free = dt[free_mask]
                dt_max = float(np.max(dt_in_free))
                dt_median = float(np.median(dt_in_free))
            else:
                dt_max = 0.0
                dt_median = 0.0

            raise RuntimeError(
                f"No valid goal cell found for target '{target_name}' at ({goal_y:.1f}, {goal_x:.1f}).\n"
                f"Diagnostics:\n"
                f"  Maze shape: ({maze_h}, {maze_w})\n"
                f"  Valid cells: {valid_count}/{total_cells} ({100*valid_count/total_cells:.1f}%)\n"
                f"  Free cells: {free_count}/{total_cells} ({100*free_count/total_cells:.1f}%)\n"
                f"  robot_radius_px: {data.robot_radius_px:.2f}\n"
                f"  dt (distance to walls) in free cells: max={dt_max:.2f}, median={dt_median:.2f}\n"
                f"  Issue: robot_radius_px ({data.robot_radius_px:.2f}) may be too large compared to dt values.\n"
                f"  Solution: Reduce robot_radius_px in config or check maze data quality."
            )

        import warnings
        gy, gx = goal_idx
        dist = ((gy - goal_y)**2 + (gx - goal_x)**2)**0.5
        warnings.warn(
            f"Target '{target_name}' at ({goal_y:.1f}, {goal_x:.1f}) is far from valid cells. "
            f"Using nearest valid cell at ({gy}, {gx}), distance: {dist:.1f} pixels."
        )

    maze_h, maze_w = costmap.maze_shape
    maze_shape = (maze_h, maze_w)
    maze_params = data.maze_params

    candidate_starts: List[Tuple[int, int]] = []

    cam_samples = _sample_start_points_in_camera(
        start_region=config.start_region,
        n_samples=config.n_start_samples,
        rng=rng,
    )

    for cam_x, cam_y in cam_samples:
        maze_y, maze_x = cam_to_maze(
            cam_x=float(cam_x),
            cam_y=float(cam_y),
            maze_params=maze_params,
            maze_shape=maze_shape,
        )
        start_idx = _nearest_valid_cell(costmap, maze_y, maze_x)
        if start_idx is not None:
            candidate_starts.append(start_idx)

    if not candidate_starts:

        if isinstance(config.start_region, CircularStartRegion):
            center_cam_x, center_cam_y = config.start_region.center
        else:                   
            center_cam_x, center_cam_y = config.start_region.center

        center_maze_y, center_maze_x = cam_to_maze(
            cam_x=float(center_cam_x),
            cam_y=float(center_cam_y),
            maze_params=maze_params,
            maze_shape=maze_shape,
        )
        fallback_start = _nearest_valid_cell(costmap, center_maze_y, center_maze_x, max_radius=200)
        if fallback_start is None:
            fallback_start = _find_nearest_valid_cell_anywhere(costmap, center_maze_y, center_maze_x)

        if fallback_start is None:
            raise RuntimeError("No valid start cells found in start region (including center fallback).")

        candidate_starts.append(fallback_start)

    def _has_synchronous_dual_arm_collision(path_maze: np.ndarray) -> bool:

        if not (
            config.enable_ump_rod_detection
            and data.ump_rod_width_maze is not None
            and data.ump_rod_length_maze is not None
        ):
            return False

        path = np.asarray(path_maze, dtype=float)
        if len(path) == 0:
            return False

        half_len = 0.5 * float(data.ump_rod_length_maze)
        tangents = np.zeros_like(path)
        for i in range(len(path)):
            if i == 0:
                d = path[1] - path[0] if len(path) > 1 else np.array([0.0, 1.0])
            elif i == len(path) - 1:
                d = path[-1] - path[-2]
            else:
                d = path[i + 1] - path[i - 1]
            norm = np.hypot(d[0], d[1])
            if norm > 1e-10:
                tangents[i] = d / norm
            else:
                tangents[i] = tangents[i - 1] if i > 0 else np.array([0.0, 1.0])

        bottom_endpoints = path - half_len * tangents
        top_endpoints = path + half_len * tangents
        has_collision, _, _ = check_dual_arm_trajectory_collision(
            bottom_endpoints=bottom_endpoints,
            top_endpoints=top_endpoints,
            rod_width_maze=float(data.ump_rod_width_maze),
            maze_shape=maze_shape,
        )
        return has_collision

    best_result: Optional[PathResult] = None
    dual_collision_rejections = 0

    for sy, sx in candidate_starts:
        result = _astar_on_grid(costmap, (sy, sx), goal_idx)
        if result is None:
            continue
        if _has_synchronous_dual_arm_collision(result.path_maze):
            dual_collision_rejections += 1
            continue
        if best_result is None or result.total_cost < best_result.total_cost:
            best_result = result

    if best_result is None:

        if isinstance(config.start_region, CircularStartRegion):
            center_cam_x, center_cam_y = config.start_region.center
        else:                   
            center_cam_x, center_cam_y = config.start_region.center

        center_maze_y, center_maze_x = cam_to_maze(
            cam_x=float(center_cam_x),
            cam_y=float(center_cam_y),
            maze_params=maze_params,
            maze_shape=maze_shape,
        )
        center_start = _nearest_valid_cell(costmap, center_maze_y, center_maze_x, max_radius=200)
        if center_start is None:
            center_start = _find_nearest_valid_cell_anywhere(costmap, center_maze_y, center_maze_x)

        if center_start is not None:
            result = _astar_on_grid(costmap, center_start, goal_idx)
            if result is not None and not _has_synchronous_dual_arm_collision(result.path_maze):
                best_result = result
            elif result is not None:
                dual_collision_rejections += 1

    if best_result is None:
        suffix = ""
        if dual_collision_rejections:
            suffix = (
                f" {dual_collision_rejections} A* candidate(s) were rejected because "
                "the two manipulators overlapped at one or more synchronous waypoints."
            )
        raise RuntimeError(
            "Failed to find any feasible collision-free path from sampled start "
            f"points or center fallback.{suffix}"
        )

    if config.enable_ump_rod_detection and data.ump_rod_width_maze is not None and data.ump_rod_length_maze is not None:

        free_mask = np.asarray(data.maze_extractor.free_mask, dtype=bool)
        maze_shape = free_mask.shape

        if data.ump_rod_length_maze is not None:
            robot_length_px = float(data.ump_rod_length_maze)
        else:
            robot_length_px = 2.0 * float(data.robot_radius_px)
        half_len = 0.5 * robot_length_px

        path_maze = np.asarray(best_result.path_maze, dtype=float)
        n = len(path_maze)

        tangents = np.zeros((n, 2), dtype=float)
        for i in range(n):
            if i == 0:
                if n > 1:
                    d = path_maze[1] - path_maze[0]
                else:
                    d = np.array([0.0, 1.0])
            elif i == n - 1:
                d = path_maze[-1] - path_maze[-2]
            else:
                d = path_maze[i + 1] - path_maze[i - 1]

            norm = np.hypot(d[0], d[1])
            if norm > 1e-10:
                tangents[i] = d / norm
            else:
                tangents[i] = tangents[i - 1] if i > 0 else np.array([0.0, 1.0])

        bottom_endpoints = path_maze - half_len * tangents
        top_endpoints = path_maze + half_len * tangents

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
                "Dual-arm collision detected in planned path; refusing to return "
                "an unsafe trajectory. "
                f"Synchronous collision waypoint(s): [{preview}] "
                f"({len(collision_steps)} total, {overlap_pixels} overlapping pixels)."
            )

        path_rod_mask = left_rod_mask | right_rod_mask

        collisions = check_path_rod_collision(
            path_maze=best_result.path_maze,
            rod_mask=path_rod_mask,
            free_mask=free_mask,
        )

        if collisions:
            import warnings
            warnings.warn(
                f"UMP rod collision with walls detected at {len(collisions)} path point(s). "
                f"Path may not be safe for execution."
            )

    return best_result
