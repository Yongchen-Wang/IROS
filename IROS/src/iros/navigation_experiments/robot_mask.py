from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..paths import ASSET_ROOT

DEFAULT_TEMPLATE = ASSET_ROOT / "robot_template.jpg"

def load_template(template_path: Optional[Path] = None):

    path = Path(template_path) if template_path else DEFAULT_TEMPLATE
    if not path.exists():
        return None, None

    bgr = cv2.imread(str(path))
    if bgr is None:
        return None, None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_connected = cv2.dilate(bin_inv, k_dilate, iterations=2)
    contours, _ = cv2.findContours(bin_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(filled, [largest], -1, 255, -1)
    M = cv2.moments(largest)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = filled.shape[1] / 2, filled.shape[0] / 2

    return filled, (cx, cy)

def _make_affine(tpl_center: Tuple[float, float], pos_x: float, pos_y: float,
                 scale: float, angle_deg: float) -> np.ndarray:

    tcx, tcy = tpl_center
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([
        [scale * cos_a, -scale * sin_a, pos_x - scale * (cos_a * tcx - sin_a * tcy)],
        [scale * sin_a, scale * cos_a, pos_y - scale * (sin_a * tcx + cos_a * tcy)],
    ], dtype=np.float64)

def warp_template(
    tpl_binary: np.ndarray,
    tpl_center: Tuple[float, float],
    pos_x: float,
    pos_y: float,
    scale: float,
    angle_deg: float,
    fw: int,
    fh: int,
    dilate_px: int = 0,
) -> np.ndarray:

    M = _make_affine(tpl_center, pos_x, pos_y, scale, angle_deg)
    warped = cv2.warpAffine(tpl_binary, M, (fw, fh), flags=cv2.INTER_NEAREST)
    if dilate_px > 0:
        ks = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        warped = cv2.dilate(warped, kernel, iterations=1)
    return warped

def compute_robot_radius_from_mask(
    scale: float = 0.075,
    dilate_px: int = 3,
    template_path: Optional[Path] = None,
    mode: str = "area_equivalent",
) -> float:

    tpl_binary, tpl_center = load_template(template_path)
    if tpl_binary is None or tpl_center is None:
        raise FileNotFoundError(f"Template not found: {template_path or DEFAULT_TEMPLATE}")

    th, tw = tpl_binary.shape[:2]
    extent = max(tw, th) * scale + dilate_px * 2
    buf_size = int(np.ceil(extent * 2)) + 10
    cx_buf, cy_buf = buf_size // 2, buf_size // 2

    warped = warp_template(
        tpl_binary, tpl_center,
        pos_x=cx_buf, pos_y=cy_buf,
        scale=scale, angle_deg=0.0,
        fw=buf_size, fh=buf_size,
        dilate_px=dilate_px,
    )

    ys, xs = np.where(warped > 0)
    if len(ys) == 0:
        raise ValueError("Warped mask is empty")

    if mode == "bounding":
        dists = np.sqrt((xs - cx_buf) ** 2 + (ys - cy_buf) ** 2)
        radius = float(np.max(dists))
    elif mode == "area_equivalent":
        area_px = float(len(ys))
        radius = float(np.sqrt(area_px / np.pi))
    elif mode == "half_length":

        max_dim = max(tw, th) * scale
        radius = max_dim / 2.0 + float(dilate_px)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'bounding', 'area_equivalent', or 'half_length'")
    return radius

def robot_radius_maze_from_mask_align(
    maze_params: Dict[str, Any],
    scale: float = 0.075,
    dilate_px: int = 3,
    template_path: Optional[Path] = None,
    radius_mode: str = "area_equivalent",
) -> float:

    maze_scale = float(maze_params.get("scale", 0.565))
    if maze_scale <= 0:
        raise ValueError("maze_params.scale must be positive")

    radius_cam = compute_robot_radius_from_mask(
        scale=scale,
        dilate_px=dilate_px,
        template_path=template_path,
        mode=radius_mode,
    )

    radius_maze = radius_cam / maze_scale
    return radius_maze
