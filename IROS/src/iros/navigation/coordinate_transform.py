from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

MazePoint = Tuple[float, float]                         
CamPoint = Tuple[float, float]                           

@dataclass
class MazeRegistration:

    cx: float
    cy: float
    scale: float
    angle_deg: float

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "MazeRegistration":
        return MazeRegistration(

            cx=float(d.get("cx", 624.0)),
            cy=float(d.get("cy", 578.0)),
            scale=float(d.get("scale", 0.565)),
            angle_deg=float(d.get("angle_deg", 0.0)),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "cx": self.cx,
            "cy": self.cy,
            "scale": self.scale,
            "angle_deg": self.angle_deg,
        }

def _build_affine(mw: float, mh: float, reg: MazeRegistration) -> np.ndarray:

    theta = np.radians(reg.angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    mx, my = mw / 2.0, mh / 2.0
    M = np.array(
        [
            [reg.scale * cos_t, -reg.scale * sin_t, reg.cx - reg.scale * (cos_t * mx - sin_t * my)],
            [reg.scale * sin_t, reg.scale * cos_t, reg.cy - reg.scale * (sin_t * mx + cos_t * my)],
        ],
        dtype=np.float64,
    )
    return M

def cam_to_maze(cam_x: float, cam_y: float, maze_params: Dict[str, float], maze_shape: Tuple[int, int]) -> MazePoint:

    mh, mw = maze_shape
    reg = MazeRegistration.from_dict(maze_params)

    M = _build_affine(mw, mh, reg)

    M_3x3 = np.vstack([M, [0.0, 0.0, 1.0]])
    M_inv = np.linalg.inv(M_3x3)

    cam_pt = np.array([cam_x, cam_y, 1.0], dtype=np.float64)
    maze_pt = M_inv @ cam_pt
    maze_x = maze_pt[0]
    maze_y = maze_pt[1]

    maze_x = float(np.clip(maze_x, 0, mw - 1))
    maze_y = float(np.clip(maze_y, 0, mh - 1))

    return maze_y, maze_x

def maze_to_cam(maze_y: float, maze_x: float, maze_params: Dict[str, float], maze_shape: Tuple[int, int]) -> CamPoint:

    mh, mw = maze_shape
    reg = MazeRegistration.from_dict(maze_params)
    M = _build_affine(mw, mh, reg)

    maze_pt = np.array([maze_x, maze_y, 1.0], dtype=np.float64)
    cam_pt = M @ maze_pt
    cam_x = float(cam_pt[0])
    cam_y = float(cam_pt[1])
    return cam_x, cam_y

def cam_segment_to_maze(
    bottom_x: float,
    bottom_y: float,
    top_x: float,
    top_y: float,
    centroid_x: float,
    centroid_y: float,
    maze_params: Dict[str, float],
    maze_shape: Tuple[int, int],
) -> Dict[str, MazePoint]:

    bottom = cam_to_maze(bottom_x, bottom_y, maze_params, maze_shape)
    top = cam_to_maze(top_x, top_y, maze_params, maze_shape)
    centroid = cam_to_maze(centroid_x, centroid_y, maze_params, maze_shape)
    return {"bottom": bottom, "top": top, "centroid": centroid}
