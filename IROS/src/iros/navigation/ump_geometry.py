from __future__ import annotations

"""
Pure geometric UMP rod utilities for Navigation V1.

These functions operate purely in maze coordinates using numpy arrays and do not
perform any image I/O or black-mask detection. They are used by the core
navigation pipeline (path planning, trajectory generation, animation).
"""

from typing import Any, Dict, List, Tuple, Optional

import numpy as np

def generate_left_rod_mask(
    bottom_endpoints: np.ndarray,
    rod_width_maze: float,
    maze_shape: Tuple[int, int],
) -> np.ndarray:

    if len(bottom_endpoints) == 0:
        return np.zeros(maze_shape, dtype=bool)

    bottom_endpoints = np.asarray(bottom_endpoints, dtype=float)
    maze_h, maze_w = maze_shape
    rod_mask = np.zeros((maze_h, maze_w), dtype=bool)

    half_width = rod_width_maze / 2.0

    for i in range(len(bottom_endpoints)):
        by, bx = bottom_endpoints[i]

        if bx > 0:

            y_min = int(np.floor(by - half_width))
            y_max = int(np.ceil(by + half_width))
            x_min = 0
            x_max = int(np.ceil(bx))

            y_min = max(0, min(y_min, maze_h - 1))
            y_max = max(0, min(y_max, maze_h - 1))
            x_min = max(0, min(x_min, maze_w - 1))
            x_max = max(0, min(x_max, maze_w - 1))

            rod_mask[y_min:y_max + 1, x_min:x_max + 1] = True
        else:

            y_min = int(np.floor(by - half_width))
            y_max = int(np.ceil(by + half_width))
            x_min = 0
            x_max = max(0, int(np.ceil(bx + rod_width_maze)))

            y_min = max(0, min(y_min, maze_h - 1))
            y_max = max(0, min(y_max, maze_h - 1))
            x_min = max(0, min(x_min, maze_w - 1))
            x_max = max(0, min(x_max, maze_w - 1))

            rod_mask[y_min:y_max + 1, x_min:x_max + 1] = True

    return rod_mask

def generate_right_rod_mask(
    top_endpoints: np.ndarray,
    rod_width_maze: float,
    maze_shape: Tuple[int, int],
) -> np.ndarray:

    if len(top_endpoints) == 0:
        return np.zeros(maze_shape, dtype=bool)

    top_endpoints = np.asarray(top_endpoints, dtype=float)
    maze_h, maze_w = maze_shape
    rod_mask = np.zeros((maze_h, maze_w), dtype=bool)

    half_width = rod_width_maze / 2.0
    right_boundary_x = float(maze_w - 1)

    for i in range(len(top_endpoints)):
        ty, tx = top_endpoints[i]

        if tx < right_boundary_x:
            y_min = int(np.floor(ty - half_width))
            y_max = int(np.ceil(ty + half_width))
            x_min = int(np.floor(tx))
            x_max = int(np.ceil(right_boundary_x))

            y_min = max(0, min(y_min, maze_h - 1))
            y_max = max(0, min(y_max, maze_h - 1))
            x_min = max(0, min(x_min, maze_w - 1))
            x_max = max(0, min(x_max, maze_w - 1))

            rod_mask[y_min:y_max + 1, x_min:x_max + 1] = True
        else:
            y_min = int(np.floor(ty - half_width))
            y_max = int(np.ceil(ty + half_width))
            x_min = max(0, int(np.floor(tx - rod_width_maze)))
            x_max = int(np.ceil(right_boundary_x))

            y_min = max(0, min(y_min, maze_h - 1))
            y_max = max(0, min(y_max, maze_h - 1))
            x_min = max(0, min(x_min, maze_w - 1))
            x_max = max(0, min(x_max, maze_w - 1))

            rod_mask[y_min:y_max + 1, x_min:x_max + 1] = True

    return rod_mask

def check_dual_arm_collision(
    left_rod_mask: np.ndarray,
    right_rod_mask: np.ndarray,
) -> Tuple[bool, np.ndarray]:

    if left_rod_mask.shape != right_rod_mask.shape:
        raise ValueError(
            f"Mask shapes must match: left={left_rod_mask.shape}, right={right_rod_mask.shape}"
        )

    overlap = left_rod_mask & right_rod_mask
    has_collision = bool(np.any(overlap))

    return has_collision, overlap

def check_dual_arm_trajectory_collision(
    bottom_endpoints: np.ndarray,
    top_endpoints: np.ndarray,
    rod_width_maze: float,
    maze_shape: Tuple[int, int],
) -> Tuple[bool, List[int], np.ndarray]:

    bottom = np.asarray(bottom_endpoints, dtype=float)
    top = np.asarray(top_endpoints, dtype=float)

    if bottom.ndim != 2 or bottom.shape[1:] != (2,):
        raise ValueError(f"bottom_endpoints must have shape [N, 2], got {bottom.shape}")
    if top.ndim != 2 or top.shape[1:] != (2,):
        raise ValueError(f"top_endpoints must have shape [N, 2], got {top.shape}")
    if len(bottom) != len(top):
        raise ValueError(
            "left/right endpoint sequences must have the same length: "
            f"left={len(bottom)}, right={len(top)}"
        )

    maze_h, maze_w = maze_shape
    overlap_union = np.zeros(maze_shape, dtype=bool)
    collision_indices: List[int] = []
    half_width = rod_width_maze / 2.0
    right_boundary_x = float(maze_w - 1)

    for i in range(len(bottom)):
        by, bx = bottom[i]
        ty, tx = top[i]

        ly0 = max(0, min(int(np.floor(by - half_width)), maze_h - 1))
        ly1 = max(0, min(int(np.ceil(by + half_width)), maze_h - 1))
        if bx > 0:
            lx0 = 0
            lx1 = max(0, min(int(np.ceil(bx)), maze_w - 1))
        else:
            lx0 = 0
            lx1 = max(0, min(max(0, int(np.ceil(bx + rod_width_maze))), maze_w - 1))

        ry0 = max(0, min(int(np.floor(ty - half_width)), maze_h - 1))
        ry1 = max(0, min(int(np.ceil(ty + half_width)), maze_h - 1))
        if tx < right_boundary_x:
            rx0 = max(0, min(int(np.floor(tx)), maze_w - 1))
            rx1 = maze_w - 1
        else:
            rx0 = max(
                0,
                min(max(0, int(np.floor(tx - rod_width_maze))), maze_w - 1),
            )
            rx1 = maze_w - 1

        oy0, oy1 = max(ly0, ry0), min(ly1, ry1)
        ox0, ox1 = max(lx0, rx0), min(lx1, rx1)
        if oy0 <= oy1 and ox0 <= ox1:
            collision_indices.append(i)
            overlap_union[oy0 : oy1 + 1, ox0 : ox1 + 1] = True

    return bool(collision_indices), collision_indices, overlap_union

def generate_dual_rod_masks(
    bottom_endpoints: np.ndarray,
    top_endpoints: np.ndarray,
    rod_width_maze: float,
    maze_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:

    left_mask = generate_left_rod_mask(bottom_endpoints, rod_width_maze, maze_shape)
    right_mask = generate_right_rod_mask(top_endpoints, rod_width_maze, maze_shape)
    return left_mask, right_mask

def generate_rod_mask_for_path(
    path_maze: np.ndarray,
    rod_width_maze: float,
    rod_length_maze: float,
    maze_shape: Tuple[int, int],
    robot_length_px: Optional[float] = None,
) -> np.ndarray:

    if len(path_maze) == 0:
        return np.zeros(maze_shape, dtype=bool)

    path_maze = np.asarray(path_maze, dtype=float)

    if robot_length_px is None:
        robot_length_px = rod_length_maze

    if robot_length_px is None or robot_length_px <= 0:
        robot_length_px = 20.0                    

    half_len = 0.5 * robot_length_px

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

    left_mask = generate_left_rod_mask(bottom_endpoints, rod_width_maze, maze_shape)
    right_mask = generate_right_rod_mask(top_endpoints, rod_width_maze, maze_shape)
    return left_mask | right_mask

def check_path_rod_collision(
    path_maze: np.ndarray,
    rod_mask: np.ndarray,
    free_mask: np.ndarray,
) -> List[Tuple[int, Dict[str, Any]]]:

    collisions: List[Tuple[int, Dict[str, Any]]] = []

    if len(path_maze) == 0:
        return collisions

    path_maze = np.asarray(path_maze, dtype=float)

    wall_mask = ~free_mask
    overlap = rod_mask & wall_mask

    if not np.any(overlap):
        return collisions

    for i in range(len(path_maze)):
        py, px = int(round(path_maze[i, 0])), int(round(path_maze[i, 1]))
        h, w = overlap.shape
        if 0 <= py < h and 0 <= px < w:
            y_min = max(0, py - 10)
            y_max = min(h - 1, py + 10)
            x_min = max(0, px - 10)
            x_max = min(w - 1, px + 10)
            if np.any(overlap[y_min:y_max + 1, x_min:x_max + 1]):
                collisions.append(
                    (
                        i,
                        {
                            "rod_region": rod_mask[y_min:y_max + 1, x_min:x_max + 1].copy(),
                            "overlap_with_walls": overlap[y_min:y_max + 1, x_min:x_max + 1].copy(),
                        },
                    )
                )

    return collisions
