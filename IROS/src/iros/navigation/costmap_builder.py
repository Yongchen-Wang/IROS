from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import NavigationConfig

CostmapArray = np.ndarray

@dataclass
class Costmap:

    costs: CostmapArray
    valid_mask: np.ndarray
    maze_shape: Tuple[int, int]

    def is_valid(self, y: int, x: int) -> bool:

        h, w = self.maze_shape
        if not (0 <= y < h and 0 <= x < w):
            return False
        return bool(self.valid_mask[y, x])

    def cost_at(self, y: int, x: int) -> float:

        if not self.is_valid(y, x):
            return float("inf")
        return float(self.costs[y, x])

def build_safety_costmap(
    maze_extractor: Any,
    robot_radius_px: float,
    config: NavigationConfig,
    ump_rod_mask: Optional[np.ndarray] = None,
    robot_length_px: Optional[float] = None,
) -> Costmap:

    if robot_radius_px <= 0:
        raise ValueError("robot_radius_px must be positive.")

    if robot_length_px is None:

        if config.ump_rod_length_maze is not None:
            robot_length_px = float(config.ump_rod_length_maze)
        else:

            robot_length_px = 2.0 * float(robot_radius_px)

    if robot_length_px <= 0:
        raise ValueError("robot_length_px must be positive.")

    override_radius = getattr(config, "effective_radius_px_override", None)
    if override_radius is not None:
        effective_radius = float(override_radius)
    else:
        effective_radius = robot_length_px / 2.0

    free_mask: np.ndarray = np.asarray(maze_extractor.free_mask, dtype=bool)
    dt: np.ndarray = np.asarray(maze_extractor.dt, dtype=float)

    if free_mask.shape != dt.shape:
        raise ValueError(
            f"free_mask shape {free_mask.shape} does not match dt shape {dt.shape}."
        )

    maze_shape: Tuple[int, int] = free_mask.shape

    valid_mask = free_mask & (dt >= effective_radius)

    if ump_rod_mask is not None:
        if ump_rod_mask.shape != maze_shape:
            raise ValueError(
                f"UMP rod mask shape {ump_rod_mask.shape} does not match maze shape {maze_shape}"
            )

        valid_mask = valid_mask & (~ump_rod_mask)

    base_cost = 1.0

    clearance = np.maximum(dt - effective_radius, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        norm_clearance = clearance / max(effective_radius, 1e-6)

        norm_clearance = np.maximum(norm_clearance, 0.0)

    use_exponential = getattr(config, "use_exponential_cost", True)
    if use_exponential:
        alpha = getattr(config, "center_preference_alpha", 2.0)
        safety_cost = config.safety_margin_weight * np.exp(-alpha * norm_clearance)
    else:

        norm_clearance_clipped = np.clip(norm_clearance, 0.0, 1.0)
        safety_cost = config.safety_margin_weight * (1.0 - norm_clearance_clipped)

    costs = np.full(maze_shape, np.inf, dtype=float)
    costs[valid_mask] = base_cost + safety_cost[valid_mask]

    if ump_rod_mask is not None:
        costs[ump_rod_mask] = np.inf

    return Costmap(costs=costs, valid_mask=valid_mask, maze_shape=maze_shape)

def costmap_to_dict(costmap: Costmap) -> Dict[str, Any]:

    return {
        "costs": costmap.costs.tolist(),
        "valid_mask": costmap.valid_mask.tolist(),
        "maze_shape": list(costmap.maze_shape),
    }

def costmap_from_dict(data: Dict[str, Any]) -> Costmap:

    costs = np.asarray(data["costs"], dtype=float)
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    maze_shape_tuple: Tuple[int, int] = tuple(data["maze_shape"])                            

    if costs.shape != valid_mask.shape or costs.shape != maze_shape_tuple:
        raise ValueError(
            "Inconsistent shapes when restoring Costmap: "
            f"costs={costs.shape}, valid_mask={valid_mask.shape}, "
            f"maze_shape={maze_shape_tuple}"
        )

    return Costmap(costs=costs, valid_mask=valid_mask, maze_shape=maze_shape_tuple)
