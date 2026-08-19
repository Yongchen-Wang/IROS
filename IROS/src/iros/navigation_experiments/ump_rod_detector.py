from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..navigation.coordinate_transform import cam_to_maze, maze_to_cam
from .robot_mask import DEFAULT_TEMPLATE, load_template

@dataclass
class UMPRodDimensions:

    rod_width_maze: float                                              
    rod_length_maze: float                                                          

def detect_ump_rods_from_images(
    dataset_dir: Path,
    black_threshold: int = 30,
    min_area: int = 100,
    left_margin: int = 50,
    right_margin: int = 50,
    max_images: int = 50,
) -> List[Dict[str, Any]]:

    ds_dir = Path(dataset_dir)

    img_dir = ds_dir / "images_vis"
    if not img_dir.exists():
        img_dir = ds_dir / "images"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {ds_dir}")

    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg')])
    if not img_files:
        raise ValueError(f"No images found in {img_dir}")

    img_files = img_files[:max_images]

    detections = []

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        detection = {
            'left_rod': None,
            'right_rod': None,
            'image_path': str(img_path),
        }

        for label_id in range(1, num_labels):                       
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            width = stats[label_id, cv2.CC_STAT_WIDTH]
            height = stats[label_id, cv2.CC_STAT_HEIGHT]

            if x < left_margin:

                rod_width = width
                rod_length = height
                detection['left_rod'] = {
                    'bbox': (x, y, width, height),
                    'area': int(area),
                    'width': float(rod_width),
                    'length': float(rod_length),
                }

            elif x + width > w - right_margin:
                rod_width = width
                rod_length = height
                detection['right_rod'] = {
                    'bbox': (x, y, width, height),
                    'area': int(area),
                    'width': float(rod_width),
                    'length': float(rod_length),
                }

        if detection['left_rod'] is not None or detection['right_rod'] is not None:
            detections.append(detection)

    return detections

def compute_rod_dimensions(
    detections: List[Dict[str, Any]],
    maze_params: Dict[str, Any],
    maze_shape: Tuple[int, int],
) -> Tuple[float, float]:

    if not detections:
        raise ValueError("No detections provided")

    widths_cam = []
    lengths_cam = []

    for det in detections:
        if det['left_rod'] is not None:
            widths_cam.append(det['left_rod']['width'])
            lengths_cam.append(det['left_rod']['length'])
        if det['right_rod'] is not None:
            widths_cam.append(det['right_rod']['width'])
            lengths_cam.append(det['right_rod']['length'])

    if not widths_cam or not lengths_cam:
        raise ValueError("No valid rod detections found")

    avg_width_cam = float(np.mean(widths_cam))
    avg_length_cam = float(np.mean(lengths_cam))

    maze_scale = float(maze_params.get("scale", 1.0))
    if maze_scale <= 0:
        raise ValueError("maze_params.scale must be positive")

    rod_width_maze = avg_width_cam / maze_scale
    rod_length_maze = avg_length_cam / maze_scale

    return rod_width_maze, rod_length_maze

def get_ideal_rod_length_from_template(
    maze_params: Dict[str, Any],
    maze_shape: Tuple[int, int],
    template_path: Optional[Path] = None,
) -> float:

    path = Path(template_path) if template_path else DEFAULT_TEMPLATE
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load template: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    rod_centers = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < 100:                           
            continue

        cx = centroids[label_id, 0]
        cy = centroids[label_id, 1]

        if cx < w * 0.2 or cx > w * 0.8:
            rod_centers.append((cx, cy))

    if len(rod_centers) < 2:

        leftmost = None
        rightmost = None
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < 100:
                continue

            x = stats[label_id, cv2.CC_STAT_LEFT]
            if leftmost is None or x < leftmost[0]:
                leftmost = (x, centroids[label_id, 1])
            if rightmost is None or x > rightmost[0]:
                rightmost = (x, centroids[label_id, 1])

        if leftmost is None or rightmost is None:
            raise ValueError("Could not find two UMP rods in template")
        rod_centers = [leftmost, rightmost]

    if len(rod_centers) < 2:
        raise ValueError("Could not find two UMP rods in template")

    rod_centers.sort(key=lambda p: p[0])
    left_rod = rod_centers[0]
    right_rod = rod_centers[-1]

    dx = right_rod[0] - left_rod[0]
    dy = right_rod[1] - left_rod[1]
    distance_cam = np.sqrt(dx * dx + dy * dy)

    maze_scale = float(maze_params.get("scale", 1.0))
    if maze_scale <= 0:
        raise ValueError("maze_params.scale must be positive")

    rod_length_maze = distance_cam / maze_scale

    return float(rod_length_maze)

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

        dx = bx - 0.0
        dy = by - by                                            

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

        dx = tx - right_boundary_x
        dy = ty - ty                                            

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
    maze_h, maze_w = maze_shape

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

    collisions = []

    if len(path_maze) == 0:
        return collisions

    path_maze = np.asarray(path_maze, dtype=float)

    wall_mask = ~free_mask
    overlap = rod_mask & wall_mask

    if np.any(overlap):

        for i in range(len(path_maze)):
            py, px = int(round(path_maze[i, 0])), int(round(path_maze[i, 1]))
            h, w = overlap.shape
            if 0 <= py < h and 0 <= px < w:

                y_min = max(0, py - 10)
                y_max = min(h - 1, py + 10)
                x_min = max(0, px - 10)
                x_max = min(w - 1, px + 10)
                if np.any(overlap[y_min:y_max + 1, x_min:x_max + 1]):
                    collisions.append((
                        i,
                        {
                            'rod_region': rod_mask[y_min:y_max + 1, x_min:x_max + 1].copy(),
                            'overlap_with_walls': overlap[y_min:y_max + 1, x_min:x_max + 1].copy(),
                        }
                    ))

    return collisions
