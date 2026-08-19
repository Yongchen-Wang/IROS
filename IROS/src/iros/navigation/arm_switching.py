from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from .path_planner import PathResult

class ArmRole(Enum):

    BOTTOM = "bottom"
    TOP = "top"

@dataclass
class EndpointTrajectories:

    path_maze: np.ndarray                             
    bottom_maze: np.ndarray                                 
    top_maze: np.ndarray                              
    tangents: np.ndarray                                              
    robot_length_px: float

@dataclass
class ArmSwitchResult:

    trajectories: EndpointTrajectories
    master_sequence: List[ArmRole]
    switch_indices: List[int] = field(default_factory=list)

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

def compute_endpoint_trajectories(
    path_result: PathResult,
    robot_length_px: float,
    initial_tangent: Optional[np.ndarray] = None,
) -> EndpointTrajectories:

    path_maze = np.asarray(path_result.path_maze, dtype=float)
    n = len(path_maze)

    if n == 0:
        raise ValueError("path_maze cannot be empty.")

    if robot_length_px <= 0:
        raise ValueError("robot_length_px must be positive.")

    half_len = 0.5 * robot_length_px
    tangents = _compute_tangents(path_maze, initial_tangent=initial_tangent)

    bottom_maze = path_maze - half_len * tangents
    top_maze = path_maze + half_len * tangents

    return EndpointTrajectories(
        path_maze=path_maze,
        bottom_maze=bottom_maze,
        top_maze=top_maze,
        tangents=tangents,
        robot_length_px=robot_length_px,
    )

def _master_at_point(
    bottom_pt: np.ndarray,
    top_pt: np.ndarray,
    goal_maze: np.ndarray,
) -> ArmRole:

    goal = np.asarray(goal_maze, dtype=float)
    bottom_dist = float(np.hypot(bottom_pt[0] - goal[0], bottom_pt[1] - goal[1]))
    top_dist = float(np.hypot(top_pt[0] - goal[0], top_pt[1] - goal[1]))

    if top_dist < bottom_dist:
        return ArmRole.TOP
    return ArmRole.BOTTOM

def compute_master_slave_sequence(
    trajectories: EndpointTrajectories,
    goal_maze: np.ndarray,
    switch_threshold: float = 0.0,
    max_switches: int = 1,
    initial_master: Optional[ArmRole] = None,
) -> List[ArmRole]:

    n = len(trajectories.path_maze)
    bottom_maze = trajectories.bottom_maze
    top_maze = trajectories.top_maze
    goal = np.asarray(goal_maze, dtype=float)

    master_seq: List[ArmRole] = []
    current_master = None
    switch_count = 0

    for i in range(n):
        bottom_pt = bottom_maze[i]
        top_pt = top_maze[i]

        bottom_dist = float(np.hypot(bottom_pt[0] - goal[0], bottom_pt[1] - goal[1]))
        top_dist = float(np.hypot(top_pt[0] - goal[0], top_pt[1] - goal[1]))

        candidate_master = ArmRole.TOP if top_dist < bottom_dist else ArmRole.BOTTOM

        if current_master is None:

            if initial_master is not None:
                current_master = initial_master
            else:
                current_master = candidate_master
        else:

            dist_diff = abs(bottom_dist - top_dist)
            if (candidate_master != current_master and 
                dist_diff >= switch_threshold and 
                switch_count < max_switches):
                current_master = candidate_master
                switch_count += 1

        master_seq.append(current_master)

    return master_seq

def find_switch_indices(master_sequence: List[ArmRole]) -> List[int]:

    switch_indices: List[int] = []
    for i in range(1, len(master_sequence)):
        if master_sequence[i] != master_sequence[i - 1]:
            switch_indices.append(i)
    return switch_indices

def _segment_intersect_2d(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> bool:

    def cross_product_orientation(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:

        return float((a[1] - o[1]) * (b[0] - o[0]) - (a[0] - o[0]) * (b[1] - o[1]))

    o1 = cross_product_orientation(p1, p2, p3)
    o2 = cross_product_orientation(p1, p2, p4)
    o3 = cross_product_orientation(p3, p4, p1)
    o4 = cross_product_orientation(p3, p4, p2)

    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True

    return False

def check_rod_collision(
    trajectories: EndpointTrajectories,
    tolerance: float = 0.0,
) -> List[Tuple[int, int]]:

    path_maze = trajectories.path_maze
    bottom_maze = trajectories.bottom_maze
    top_maze = trajectories.top_maze

    collisions: List[Tuple[int, int]] = []
    n = len(path_maze)

    for i in range(n - 1):

        rod1_start = bottom_maze[i]
        rod1_end = top_maze[i]

        rod2_start = bottom_maze[i + 1]
        rod2_end = top_maze[i + 1]

        if _segment_intersect_2d(rod1_start, rod1_end, rod2_start, rod2_end):
            collisions.append((i, i + 1))

    return collisions

def compute_arm_switching(
    path_result: PathResult,
    robot_length_px: float,
    switch_threshold: float = 0.0,
    max_switches: int = 1,
    initial_master: Optional[ArmRole] = None,
    initial_tangent: Optional[np.ndarray] = None,
) -> ArmSwitchResult:

    trajectories = compute_endpoint_trajectories(
        path_result=path_result,
        robot_length_px=robot_length_px,
        initial_tangent=initial_tangent,
    )
    goal_maze = np.asarray(path_result.goal_maze, dtype=float)
    master_sequence = compute_master_slave_sequence(
        trajectories, 
        goal_maze,
        switch_threshold=switch_threshold,
        max_switches=max_switches,
        initial_master=initial_master,
    )
    switch_indices = find_switch_indices(master_sequence)

    return ArmSwitchResult(
        trajectories=trajectories,
        master_sequence=master_sequence,
        switch_indices=switch_indices,
    )
