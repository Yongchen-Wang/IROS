#!/usr/bin/env python3

import os
import sys
import shutil
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
import gradio as gr
import traceback
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from sam2.build_sam import build_sam2_video_predictor
from iros.paths import ASSET_ROOT, CHECKPOINT_ROOT, DATA_ROOT, FRAME_ROOT, OUTPUT_ROOT

BASE_DIR = str(FRAME_ROOT)
DEFAULT_TEMPLATE = str(ASSET_ROOT / "robot_template.jpg")
YOLO_DATASET_DIR = str(OUTPUT_ROOT / "yolo_datasets")
SAM2_DATASET_DIR = str(DATA_ROOT)
SAM2_CHECKPOINT = str(CHECKPOINT_ROOT / "sam2.1_hiera_tiny.pt")
MODEL_CFG = 'configs/sam2.1/sam2.1_hiera_t.yaml'

_predictor = None
_device = None

def get_device():
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _device = torch.device('mps')
        else:
            _device = torch.device('cpu')
    return _device

def get_predictor():
    global _predictor
    if _predictor is None:
        device = get_device()
        print(f'[SAM2] Loading model on {device} ...')
        _predictor = build_sam2_video_predictor(MODEL_CFG, SAM2_CHECKPOINT, device=device)
        print('[SAM2] Model loaded.')
    return _predictor

def list_video_dirs():
    if not os.path.isdir(BASE_DIR):
        return []
    dirs = []
    for d in sorted(os.listdir(BASE_DIR)):
        full = os.path.join(BASE_DIR, d)
        if os.path.isdir(full):
            jpgs = [f for f in os.listdir(full) if f.lower().endswith(('.jpg', '.jpeg'))]
            if jpgs:
                dirs.append(d)
    return dirs

def get_sorted_frames(video_dir_name):
    d = os.path.join(BASE_DIR, video_dir_name)
    names = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg'))]
    names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return names

def load_template(template_path):

    bgr = cv2.imread(template_path)
    if bgr is None:
        return None, None, None, None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_connected = cv2.dilate(bin_inv, k_dilate, iterations=2)
    contours, _ = cv2.findContours(bin_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bgr, bin_inv, None, None
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(filled, [largest], -1, 255, -1)
    M = cv2.moments(largest)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = filled.shape[1] / 2, filled.shape[0] / 2
    return bgr, filled, largest, (cx, cy)

def find_endpoint_circles(mask_uint8):

    if mask_uint8 is None or np.sum(mask_uint8 > 0) < 100:
        return None

    dist_map = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    max_r = float(dist_map.max())
    if max_r < 2:
        return None

    fg_ys, fg_xs = np.where(mask_uint8 > 0)
    extent = max(float(fg_xs.max() - fg_xs.min()),
                 float(fg_ys.max() - fg_ys.min()), 1.0)

    ks = max(3, int(extent * 0.05)) | 1
    dilated = cv2.dilate(dist_map, np.ones((ks, ks)))

    thresh = max_r * 0.20
    local_max = (dist_map == dilated) & (dist_map >= thresh)

    ys_peak, xs_peak = np.where(local_max)
    if len(xs_peak) < 2:

        ks2 = max(3, ks // 2) | 1
        dilated2 = cv2.dilate(dist_map, np.ones((ks2, ks2)))
        local_max = (dist_map == dilated2) & (dist_map >= thresh)
        ys_peak, xs_peak = np.where(local_max)
        if len(xs_peak) < 2:
            return None

    radii = dist_map[ys_peak, xs_peak]

    n_cand = min(80, len(xs_peak))
    top_idx = np.argsort(-radii)[:n_cand]

    best_d2 = 0.0
    bi, bj = top_idx[0], top_idx[min(1, len(top_idx) - 1)]
    for i in range(len(top_idx)):
        xi, yi = float(xs_peak[top_idx[i]]), float(ys_peak[top_idx[i]])
        for j in range(i + 1, len(top_idx)):
            xj, yj = float(xs_peak[top_idx[j]]), float(ys_peak[top_idx[j]])
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            if d2 > best_d2:
                best_d2 = d2
                bi, bj = top_idx[i], top_idx[j]

    x1, y1, r1 = float(xs_peak[bi]), float(ys_peak[bi]), float(radii[bi])
    x2, y2, r2 = float(xs_peak[bj]), float(ys_peak[bj]), float(radii[bj])

    dd = np.sqrt(best_d2)
    if dd < 5:
        return None

    refined = []
    for xi, yi, ri in [(x1, y1, r1), (x2, y2, r2)]:
        refine_r = max(int(ri * 0.5), 3)
        y_lo = max(0, int(yi) - refine_r)
        y_hi = min(dist_map.shape[0], int(yi) + refine_r + 1)
        x_lo = max(0, int(xi) - refine_r)
        x_hi = min(dist_map.shape[1], int(xi) + refine_r + 1)
        patch = dist_map[y_lo:y_hi, x_lo:x_hi]
        weight = np.maximum(patch - ri * 0.8, 0)
        total_w = weight.sum()
        if total_w > 0:
            ys_l, xs_l = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
            rx = x_lo + float(np.sum(xs_l * weight) / total_w)
            ry = y_lo + float(np.sum(ys_l * weight) / total_w)
            refined.append((rx, ry, ri))
        else:
            refined.append((xi, yi, ri))
    x1, y1, r1 = refined[0]
    x2, y2, r2 = refined[1]

    dd = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

    return {
        "p1": (x1, y1), "r1": r1,
        "p2": (x2, y2), "r2": r2,
        "midpoint": ((x1 + x2) / 2, (y1 + y2) / 2),
        "dist": dd,
        "angle_deg": angle,
    }

def get_template_endpoints(tpl_binary):

    return find_endpoint_circles(tpl_binary)

def find_bottom_tpl_key(tpl_ep, tpl_center, ref_pos_x, ref_pos_y, ref_scale, ref_angle_deg):

    if tpl_ep is None:
        return "p1"
    M = make_affine(tpl_center, ref_pos_x, ref_pos_y, ref_scale, ref_angle_deg)
    p1_frame = M @ np.array([tpl_ep["p1"][0], tpl_ep["p1"][1], 1.0])
    p2_frame = M @ np.array([tpl_ep["p2"][0], tpl_ep["p2"][1], 1.0])
    return "p1" if p1_frame[1] >= p2_frame[1] else "p2"

def align_from_anchor(tpl_ep, tpl_center, anchor_key, sam_anchor_pos,
                      scale, angle_deg):

    tpl_anchor = tpl_ep[anchor_key]
    tpl_cx, tpl_cy = tpl_center

    dx = tpl_cx - tpl_anchor[0]
    dy = tpl_cy - tpl_anchor[1]

    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    dx_frame = scale * (cos_a * dx - sin_a * dy)
    dy_frame = scale * (sin_a * dx + cos_a * dy)

    pos_x = sam_anchor_pos[0] + dx_frame
    pos_y = sam_anchor_pos[1] + dy_frame

    return {
        "pos_x": float(pos_x), "pos_y": float(pos_y),
        "scale": scale, "angle_deg": angle_deg,
        "method": f"anchor_{anchor_key}",
    }

def compute_angle_from_endpoints(sam_bottom_pos, sam_top_pos,
                                  tpl_ep, bottom_tpl_key):

    top_tpl_key = "p2" if bottom_tpl_key == "p1" else "p1"

    dx_f = sam_top_pos[0] - sam_bottom_pos[0]
    dy_f = sam_top_pos[1] - sam_bottom_pos[1]

    tpl_b = tpl_ep[bottom_tpl_key]
    tpl_t = tpl_ep[top_tpl_key]
    dx_t = tpl_t[0] - tpl_b[0]
    dy_t = tpl_t[1] - tpl_b[1]

    sam_angle = np.degrees(np.arctan2(dy_f, dx_f))
    tpl_angle = np.degrees(np.arctan2(dy_t, dx_t))
    return sam_angle - tpl_angle

_cache = {}

def _get_cached(video_dir_name, template_path):
    key = (video_dir_name, template_path)
    if key not in _cache:
        tpl_bgr, tpl_binary, tpl_contour, tpl_center = load_template(template_path)
        tpl_endpoints = get_template_endpoints(tpl_binary) if tpl_binary is not None else None
        frames = get_sorted_frames(video_dir_name)
        frame_path = os.path.join(BASE_DIR, video_dir_name, frames[0])
        frame_bgr = cv2.imread(frame_path)
        _cache[key] = {
            "tpl_bgr": tpl_bgr, "tpl_binary": tpl_binary,
            "tpl_contour": tpl_contour, "tpl_center": tpl_center,
            "tpl_endpoints": tpl_endpoints,
            "frame_bgr": frame_bgr, "frames": frames,
        }
    return _cache[key]

def get_mask_pose(mask_uint8):

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 10:
        return None
    M = cv2.moments(largest)
    if M["m00"] < 1:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    mu20 = M["mu20"]
    mu02 = M["mu02"]
    mu11 = M["mu11"]
    angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

    rect = cv2.minAreaRect(largest)
    rect_w, rect_h = rect[1]

    return {
        "cx": cx, "cy": cy,
        "angle_rad": angle_rad,
        "angle_deg": np.degrees(angle_rad),
        "area": area,
        "rect_w": rect_w, "rect_h": rect_h,
    }

def make_affine(tpl_center, pos_x, pos_y, scale, angle_deg):

    tcx, tcy = tpl_center
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([
        [scale * cos_a, -scale * sin_a, pos_x - scale * (cos_a * tcx - sin_a * tcy)],
        [scale * sin_a,  scale * cos_a, pos_y - scale * (sin_a * tcx + cos_a * tcy)],
    ], dtype=np.float64)

def warp_template(tpl_binary, tpl_center, pos_x, pos_y, scale, angle_deg, fw, fh,
                   dilate_px=0):

    M = make_affine(tpl_center, pos_x, pos_y, scale, angle_deg)
    warped = cv2.warpAffine(tpl_binary, M, (fw, fh), flags=cv2.INTER_NEAREST)
    if dilate_px > 0:
        ks = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        warped = cv2.dilate(warped, kernel, iterations=1)
    return warped

def complete_single_mask(sam_mask_uint8, tpl_binary, tpl_center,
                         ref_scale, ref_angle_deg,
                         tpl_endpoints=None, use_endpoint_matching=True,
                         bottom_tpl_key="p1",
                         prev_bottom_pos=None, prev_top_pos=None,
                         last_good_angle=None,
                         dilate_px=0):

    fh, fw = sam_mask_uint8.shape[:2]

    if np.sum(sam_mask_uint8 > 0) < 50:
        return sam_mask_uint8, np.zeros_like(sam_mask_uint8), None

    top_tpl_key = "p2" if bottom_tpl_key == "p1" else "p1"
    used_angle = last_good_angle if last_good_angle is not None else ref_angle_deg
    current_bottom_pos = None
    current_top_pos = None
    align = None

    if use_endpoint_matching and tpl_endpoints is not None:
        sam_ep = find_endpoint_circles(sam_mask_uint8)

        if sam_ep is not None:
            sp1 = np.array(sam_ep["p1"])
            sp2 = np.array(sam_ep["p2"])

            if prev_top_pos is not None:

                prev_t = np.array(prev_top_pos)
                d1 = np.linalg.norm(sp1 - prev_t)
                d2 = np.linalg.norm(sp2 - prev_t)
                if d1 <= d2:
                    sam_top = (float(sp1[0]), float(sp1[1]))
                    sam_bottom = (float(sp2[0]), float(sp2[1]))
                else:
                    sam_top = (float(sp2[0]), float(sp2[1]))
                    sam_bottom = (float(sp1[0]), float(sp1[1]))
            else:

                if sp1[1] >= sp2[1]:
                    sam_bottom = (float(sp1[0]), float(sp1[1]))
                    sam_top = (float(sp2[0]), float(sp2[1]))
                else:
                    sam_bottom = (float(sp2[0]), float(sp2[1]))
                    sam_top = (float(sp1[0]), float(sp1[1]))

            tentative_angle = compute_angle_from_endpoints(
                sam_bottom, sam_top, tpl_endpoints, bottom_tpl_key)
            angle_diff = abs(((tentative_angle - used_angle) + 180) % 360 - 180)
            if angle_diff > 90:

                sam_bottom, sam_top = sam_top, sam_bottom
                tentative_angle = compute_angle_from_endpoints(
                    sam_bottom, sam_top, tpl_endpoints, bottom_tpl_key)

            current_bottom_pos = sam_bottom
            current_top_pos = sam_top

            expected_dist = tpl_endpoints["dist"] * ref_scale
            actual_dist = np.linalg.norm(
                np.array(sam_bottom) - np.array(sam_top))
            dist_ratio = actual_dist / max(expected_dist, 1)

            if 0.65 < dist_ratio < 1.5:

                used_angle = tentative_angle
                align = align_from_anchor(
                    tpl_endpoints, tpl_center,
                    top_tpl_key, sam_top,
                    ref_scale, used_angle)
                align["method"] = f"dual_endpoint_top_anchor(ratio={dist_ratio:.2f})"
            else:

                align = align_from_anchor(
                    tpl_endpoints, tpl_center,
                    top_tpl_key, sam_top,
                    ref_scale, used_angle)
                align["method"] = f"top_anchor(ratio={dist_ratio:.2f})"

    if align is None:
        pose = get_mask_pose(sam_mask_uint8)
        if pose is not None:
            align = {
                "pos_x": pose["cx"], "pos_y": pose["cy"],
                "scale": ref_scale, "angle_deg": used_angle,
                "method": "centroid_fallback",
            }
        else:
            return sam_mask_uint8, np.zeros_like(sam_mask_uint8), None

    warped = warp_template(tpl_binary, tpl_center,
                           align["pos_x"], align["pos_y"],
                           align["scale"], align["angle_deg"], fw, fh,
                           dilate_px=dilate_px)

    complete = np.maximum(sam_mask_uint8, warped)

    tracking = {
        "pos": (align["pos_x"], align["pos_y"]),
        "scale": align["scale"],
        "angle_deg": align["angle_deg"],
        "method": align["method"],
        "bottom_pos": current_bottom_pos,
        "top_pos": current_top_pos,
    }

    return complete, warped, tracking

def build_overlay(frame_bgr, tpl_binary, tpl_contour, tpl_center,
                  pos_x, pos_y, scale, angle_deg, alpha, dilate_px=0):

    fh, fw = frame_bgr.shape[:2]
    M = make_affine(tpl_center, pos_x, pos_y, scale, angle_deg)
    warped_mask = cv2.warpAffine(tpl_binary, M, (fw, fh), flags=cv2.INTER_NEAREST)
    if dilate_px > 0:
        ks = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        warped_mask = cv2.dilate(warped_mask, kernel, iterations=1)

    warped_contour = None
    if tpl_contour is not None:
        pts = tpl_contour.reshape(-1, 2).astype(np.float64)
        ones = np.ones((pts.shape[0], 1))
        pts_h = np.hstack([pts, ones])
        warped_pts = (M @ pts_h.T).T
        warped_contour = warped_pts.reshape(-1, 1, 2).astype(np.int32)

    vis = frame_bgr.copy()
    overlay_color = np.zeros_like(vis)
    overlay_color[:] = (0, 200, 200)
    mask_3c = (warped_mask > 0).astype(np.float32)[..., None]
    vis = (vis * (1 - mask_3c * alpha) + overlay_color * mask_3c * alpha).astype(np.uint8)
    if warped_contour is not None:
        cv2.drawContours(vis, [warped_contour], -1, (0, 255, 0), 2)
    cx_i, cy_i = int(pos_x), int(pos_y)
    cv2.drawMarker(vis, (cx_i, cy_i), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    warped_area = int(np.sum(warped_mask > 0))
    return vis_rgb, warped_mask, warped_area

def build_completion_vis(frame_bgr, sam_mask, complete_mask, warped_tpl):

    vis = frame_bgr.copy()
    h, w = vis.shape[:2]

    sam_2d = np.squeeze(sam_mask)
    if sam_2d.shape[:2] != (h, w):
        sam_2d = cv2.resize(sam_2d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    sam_only = (sam_2d > 0).astype(np.uint8)

    tpl_only = ((warped_tpl > 0) & (sam_2d == 0)).astype(np.uint8)

    blue_overlay = np.zeros_like(vis)
    blue_overlay[:] = (200, 100, 0)       
    sam_3c = sam_only[..., None].astype(np.float32)
    vis = (vis * (1 - sam_3c * 0.4) + blue_overlay * sam_3c * 0.4).astype(np.uint8)

    yellow_overlay = np.zeros_like(vis)
    yellow_overlay[:] = (0, 200, 200)       
    tpl_3c = tpl_only[..., None].astype(np.float32)
    vis = (vis * (1 - tpl_3c * 0.5) + yellow_overlay * tpl_3c * 0.5).astype(np.uint8)

    comp_2d = (complete_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

def build_template_only_vis(frame_bgr, warped_tpl):

    vis = frame_bgr.copy()
    h, w = vis.shape[:2]

    tpl_2d = np.squeeze(warped_tpl)
    if tpl_2d.shape[:2] != (h, w):
        tpl_2d = cv2.resize(tpl_2d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    tpl_only = (tpl_2d > 0).astype(np.uint8)

    yellow_overlay = np.zeros_like(vis)
    yellow_overlay[:] = (0, 200, 200)       
    tpl_3c = tpl_only[..., None].astype(np.float32)
    vis = (vis * (1 - tpl_3c * 0.45) + yellow_overlay * tpl_3c * 0.45).astype(np.uint8)

    contours, _ = cv2.findContours((tpl_only * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

def mask_to_yolo_seg(mask, img_w, img_h, simplify_tolerance=2.0):

    mask_2d = np.squeeze(mask)
    if mask_2d.ndim != 2:
        return None
    mask_uint8 = (mask_2d > 0).astype(np.uint8) * 255
    mh, mw = mask_2d.shape
    if mh != img_h or mw != img_w:
        mask_uint8 = cv2.resize(mask_uint8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if simplify_tolerance > 0:
        epsilon = simplify_tolerance * cv2.arcLength(largest, True) / 100.0
        largest = cv2.approxPolyDP(largest, epsilon, True)
    if len(largest) < 3:
        return None
    polygon = largest.reshape(-1, 2).astype(np.float32)
    polygon[:, 0] /= img_w
    polygon[:, 1] /= img_h
    return polygon.flatten().tolist()

def on_load_data(video_dir_name, template_path):
    global _cache
    _cache.clear()
    if not video_dir_name:
        return None, None, None, None, gr.update(), "请先选择视频目录"
    if not template_path or not os.path.exists(template_path):
        return None, None, None, None, gr.update(), f"模板文件不存在: {template_path}"
    try:
        data = _get_cached(video_dir_name, template_path)
        tpl_bgr, tpl_binary, tpl_contour, tpl_center = (
            data["tpl_bgr"], data["tpl_binary"], data["tpl_contour"], data["tpl_center"])
        frame_bgr, frames = data["frame_bgr"], data["frames"]
        if tpl_bgr is None:
            return None, None, None, None, gr.update(), "模板图片读取失败"
        fh, fw = frame_bgr.shape[:2]
        th, tw = tpl_bgr.shape[:2]
        tpl_vis = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2RGB).copy()
        if tpl_contour is not None:
            cv2.drawContours(tpl_vis, [tpl_contour], -1, (0, 255, 0), 4)
            cv2.drawMarker(tpl_vis, (int(tpl_center[0]), int(tpl_center[1])),
                           (255, 0, 0), cv2.MARKER_CROSS, 30, 3)
        init_x, init_y = float(fw / 2), float(fh / 2)
        init_scale = 0.075
        vis_rgb, _, _ = build_overlay(
            frame_bgr, tpl_binary, tpl_contour, tpl_center,
            init_x, init_y, init_scale, 0.0, 0.4)
        info = (f"✅ 加载成功! 视频: {video_dir_name} ({len(frames)}帧, {fw}x{fh}), "
                f"模板: {tw}x{th}\n\n👆 点击预览图定位 → 滑条调缩放/旋转")
        return tpl_vis, vis_rgb, init_x, init_y, gr.update(value=init_scale), info
    except Exception as e:
        traceback.print_exc()
        return None, None, None, None, gr.update(), f"加载失败: {e}"

def on_click_frame(pos_x, pos_y, video_dir_name, template_path,
                   scale, angle_deg, dilate_px, alpha, evt: gr.SelectData):
    x, y = float(evt.index[0]), float(evt.index[1])
    try:
        data = _get_cached(video_dir_name, template_path)
        vis_rgb, _, warped_area = build_overlay(
            data["frame_bgr"], data["tpl_binary"], data["tpl_contour"],
            data["tpl_center"], x, y, scale, angle_deg, alpha / 100.0,
            dilate_px=int(dilate_px))
        info = f"📍 ({x:.0f}, {y:.0f}), 缩放: {scale:.3f}, 旋转: {angle_deg:.1f}°, 膨胀: {int(dilate_px)}px, 覆盖: {warped_area}px²"
        return vis_rgb, x, y, info
    except Exception as e:
        return None, pos_x, pos_y, f"错误: {e}"

def on_slider_change(pos_x, pos_y, video_dir_name, template_path,
                     scale, angle_deg, dilate_px, alpha):
    if not video_dir_name or not template_path:
        return None, "请先加载数据"
    if pos_x is None or pos_y is None:
        return None, "请先点击预览图设置位置"
    try:
        data = _get_cached(video_dir_name, template_path)
        vis_rgb, _, warped_area = build_overlay(
            data["frame_bgr"], data["tpl_binary"], data["tpl_contour"],
            data["tpl_center"], float(pos_x), float(pos_y),
            scale, angle_deg, alpha / 100.0, dilate_px=int(dilate_px))
        info = f"📍 ({pos_x:.0f}, {pos_y:.0f}), 缩放: {scale:.3f}, 旋转: {angle_deg:.1f}°, 膨胀: {int(dilate_px)}px, 覆盖: {warped_area}px²"
        return vis_rgb, info
    except Exception as e:
        return None, f"错误: {e}"

def on_save_params(pos_x, pos_y, video_dir_name, template_path, scale, angle_deg):
    if not video_dir_name or not template_path:
        return "请先加载数据"
    if pos_x is None or pos_y is None:
        return "请先点击预览图设置位置"
    try:
        pos_x, pos_y = float(pos_x), float(pos_y)
        data = _get_cached(video_dir_name, template_path)
        tpl_center = data["tpl_center"]
        fh, fw = data["frame_bgr"].shape[:2]
        M = make_affine(tpl_center, pos_x, pos_y, scale, angle_deg)
        out_dir = str(OUTPUT_ROOT / "template_alignment")
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, "align_params.npz"), affine_M=M,
                 pos_x=pos_x, pos_y=pos_y, scale=float(scale),
                 angle_deg=float(angle_deg),
                 tpl_center_x=tpl_center[0], tpl_center_y=tpl_center[1])
        warped_mask = cv2.warpAffine(data["tpl_binary"], M, (fw, fh), flags=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_dir, "template_mask_frame0.png"), warped_mask)
        return (f"✅ 已保存! 位置:({pos_x:.0f},{pos_y:.0f}), "
                f"缩放:{scale:.4f}, 旋转:{angle_deg:.1f}°\n"
                f"📁 {out_dir}")
    except Exception as e:
        traceback.print_exc()
        return f"保存失败: {e}"

_propagation_results = {}

def on_s2_load_frame(video_dir_name, frame_idx):

    if not video_dir_name:
        return None, "请先在 Step 1 选择视频目录"
    frames = get_sorted_frames(video_dir_name)
    if not frames:
        return None, "无帧"
    idx = max(0, min(int(frame_idx), len(frames) - 1))
    path = os.path.join(BASE_DIR, video_dir_name, frames[idx])
    img = Image.open(path).convert('RGB')
    return img, f"帧 {idx}/{len(frames)-1}, 尺寸: {img.size[0]}x{img.size[1]}"

def on_s2_click(video_dir_name, frame_idx, point_type,
                points_state, labels_state, evt: gr.SelectData):

    if not video_dir_name:
        return None, points_state, labels_state, "请选择视频目录"
    x, y = evt.index[0], evt.index[1]
    label = 1 if point_type == "正样本 (前景)" else 0
    points_state.append([x, y])
    labels_state.append(label)

    frames = get_sorted_frames(video_dir_name)
    idx = max(0, min(int(frame_idx), len(frames) - 1))
    img = Image.open(os.path.join(BASE_DIR, video_dir_name, frames[idx])).convert('RGB')
    draw = ImageDraw.Draw(img)
    for i, (pt, lbl) in enumerate(zip(points_state, labels_state)):
        c = (0, 255, 0) if lbl == 1 else (255, 0, 0)
        r = 6
        draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r], fill=c, outline='white', width=2)
        draw.text((pt[0]+r+4, pt[1]-r), str(i+1), fill='white')
    info = f"已选 {len(points_state)} 个点"
    return img, points_state, labels_state, info

def on_s2_clear(video_dir_name, frame_idx):

    if video_dir_name:
        frames = get_sorted_frames(video_dir_name)
        idx = max(0, min(int(frame_idx), len(frames) - 1))
        img = Image.open(os.path.join(BASE_DIR, video_dir_name, frames[idx])).convert('RGB')
        return img, [], [], "已清除"
    return None, [], [], "已清除"

def on_s2_preview_mask(video_dir_name, template_path, frame_idx,
                       points_state, labels_state,
                       pos_x, pos_y, scale, angle_deg):

    if not video_dir_name:
        return None, "请先选择视频目录"
    if not points_state:
        return None, "请先在图上点击标注至少一个点"

    try:

        predictor = get_predictor()
        video_path = os.path.join(BASE_DIR, video_dir_name)
        inference_state = predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

        frame_idx = int(frame_idx)
        points_np = np.array(points_state, dtype=np.float32)
        labels_np = np.array(labels_state, dtype=np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx, obj_id=1,
            points=points_np, labels=labels_np)

        sam_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
        predictor.reset_state(inference_state)

        frames = get_sorted_frames(video_dir_name)
        idx = max(0, min(frame_idx, len(frames) - 1))
        frame_bgr = cv2.imread(os.path.join(video_path, frames[idx]))
        fh, fw = frame_bgr.shape[:2]

        sam_2d = np.squeeze(sam_mask).astype(np.uint8)
        if sam_2d.shape[:2] != (fh, fw):
            sam_2d = cv2.resize(sam_2d, (fw, fh), interpolation=cv2.INTER_NEAREST)
        sam_2d = (sam_2d > 0).astype(np.uint8) * 255

        sam_pose = get_mask_pose(sam_2d)

        sam_ep = find_endpoint_circles(sam_2d)

        vis = frame_bgr.copy()

        blue_overlay = np.zeros_like(vis)
        blue_overlay[:] = (200, 100, 0)
        sam_3c = (sam_2d > 0).astype(np.float32)[..., None]
        vis = (vis * (1 - sam_3c * 0.4) + blue_overlay * sam_3c * 0.4).astype(np.uint8)

        sam_contours, _ = cv2.findContours(sam_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, sam_contours, -1, (0, 0, 255), 2)

        if sam_pose:
            scx, scy = int(sam_pose["cx"]), int(sam_pose["cy"])
            cv2.drawMarker(vis, (scx, scy), (200, 200, 0), cv2.MARKER_CROSS, 15, 1)
            cv2.putText(vis, "CEN", (scx + 10, scy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

        sam_bottom = None
        sam_top = None
        bottom_tpl_key = None

        if sam_ep:

            sp1 = np.array(sam_ep["p1"])
            sp2 = np.array(sam_ep["p2"])
            if sp1[1] >= sp2[1]:
                sam_bottom = (float(sp1[0]), float(sp1[1]))
                sam_top = (float(sp2[0]), float(sp2[1]))
                sam_bottom_r = sam_ep["r1"]
                sam_top_r = sam_ep["r2"]
            else:
                sam_bottom = (float(sp2[0]), float(sp2[1]))
                sam_top = (float(sp1[0]), float(sp1[1]))
                sam_bottom_r = sam_ep["r2"]
                sam_top_r = sam_ep["r1"]

            bx, by = int(sam_bottom[0]), int(sam_bottom[1])
            cv2.circle(vis, (bx, by), int(sam_bottom_r), (0, 255, 0), 3)
            cv2.circle(vis, (bx, by), 5, (0, 255, 0), -1)
            cv2.putText(vis, "BOTTOM", (bx + int(sam_bottom_r) + 5, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            tx, ty = int(sam_top[0]), int(sam_top[1])
            cv2.circle(vis, (tx, ty), int(sam_top_r), (0, 255, 255), 2)
            cv2.circle(vis, (tx, ty), 4, (0, 255, 255), -1)
            cv2.putText(vis, "TOP", (tx + int(sam_top_r) + 5, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.line(vis, (bx, by), (tx, ty), (0, 255, 255), 1, cv2.LINE_AA)

        info_lines = []
        info_lines.append(f"━━━ SAM Mask ━━━")
        info_lines.append(f"像素: {int(np.sum(sam_2d > 0))}")
        if sam_pose:
            info_lines.append(f"质心: ({sam_pose['cx']:.0f}, {sam_pose['cy']:.0f})")

        info_lines.append(f"\n━━━ 端点检测 ━━━")
        if sam_ep and sam_bottom:
            info_lines.append(f"✅ 检测到两端圆形!")
            info_lines.append(f"🔽 BOTTOM: ({sam_bottom[0]:.0f},{sam_bottom[1]:.0f}), r={sam_bottom_r:.1f}")
            info_lines.append(f"🔼 TOP: ({sam_top[0]:.0f},{sam_top[1]:.0f}), r={sam_top_r:.1f}")
            info_lines.append(f"端点距离: {sam_ep['dist']:.1f}px")
            info_lines.append(f"端点角度: {sam_ep['angle_deg']:.1f}°")
        else:
            info_lines.append("❌ 未检测到端点圆")

        if pos_x is not None and pos_y is not None and template_path:
            pos_x, pos_y = float(pos_x), float(pos_y)
            data = _get_cached(video_dir_name, template_path)
            tpl_binary = data["tpl_binary"]
            tpl_contour = data["tpl_contour"]
            tpl_center = data["tpl_center"]
            tpl_endpoints = data.get("tpl_endpoints")

            if tpl_binary is not None:

                M_s1 = make_affine(tpl_center, pos_x, pos_y, scale, angle_deg)
                warped_s1 = cv2.warpAffine(tpl_binary, M_s1, (fw, fh),
                                           flags=cv2.INTER_NEAREST)
                if tpl_contour is not None:
                    pts = tpl_contour.reshape(-1, 2).astype(np.float64)
                    ones = np.ones((pts.shape[0], 1))
                    pts_h = np.hstack([pts, ones])
                    wc_s1 = (M_s1 @ pts_h.T).T.reshape(-1, 1, 2).astype(np.int32)
                    cv2.drawContours(vis, [wc_s1], -1, (0, 255, 0), 2)

                iou_s1 = int(np.sum((sam_2d > 0) & (warped_s1 > 0))) / max(
                    int(np.sum((sam_2d > 0) | (warped_s1 > 0))), 1)

                info_lines.append(f"\n━━━ Step1 对齐 (绿) ━━━")
                info_lines.append(f"中心: ({pos_x:.0f},{pos_y:.0f})")
                info_lines.append(f"缩放: {scale:.4f}, 角度: {angle_deg:.1f}°")
                info_lines.append(f"IoU: {iou_s1:.1%}")

                if tpl_endpoints:
                    bottom_tpl_key = find_bottom_tpl_key(
                        tpl_endpoints, tpl_center, pos_x, pos_y, scale, angle_deg)
                    top_tpl_key = "p2" if bottom_tpl_key == "p1" else "p1"

                    p_b_frame = M_s1 @ np.array([
                        tpl_endpoints[bottom_tpl_key][0],
                        tpl_endpoints[bottom_tpl_key][1], 1.0])
                    p_t_frame = M_s1 @ np.array([
                        tpl_endpoints[top_tpl_key][0],
                        tpl_endpoints[top_tpl_key][1], 1.0])

                    cv2.drawMarker(vis,
                                   (int(p_b_frame[0]), int(p_b_frame[1])),
                                   (0, 0, 255), cv2.MARKER_TRIANGLE_DOWN, 15, 2)
                    cv2.putText(vis, "TPL_B",
                                (int(p_b_frame[0]) + 10, int(p_b_frame[1]) + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                    info_lines.append(f"模板底部端点({bottom_tpl_key}): "
                                      f"({p_b_frame[0]:.0f},{p_b_frame[1]:.0f})")

                if sam_ep and sam_bottom and tpl_endpoints and bottom_tpl_key:

                    expected_dist = tpl_endpoints["dist"] * scale
                    actual_dist = np.linalg.norm(
                        np.array(sam_bottom) - np.array(sam_top))
                    dist_ratio = actual_dist / max(expected_dist, 1)

                    if 0.65 < dist_ratio < 1.5:

                        anchor_angle = compute_angle_from_endpoints(
                            sam_bottom, sam_top, tpl_endpoints, bottom_tpl_key)
                        anchor_align = align_from_anchor(
                            tpl_endpoints, tpl_center,
                            bottom_tpl_key, sam_bottom,
                            scale, anchor_angle)
                        method_str = f"dual_endpoint(ratio={dist_ratio:.2f})"
                    else:

                        anchor_align = align_from_anchor(
                            tpl_endpoints, tpl_center,
                            bottom_tpl_key, sam_bottom,
                            scale, angle_deg)
                        method_str = f"bottom_anchor(ratio={dist_ratio:.2f})"

                    M_anchor = make_affine(tpl_center,
                                           anchor_align["pos_x"],
                                           anchor_align["pos_y"],
                                           anchor_align["scale"],
                                           anchor_align["angle_deg"])
                    warped_anchor = cv2.warpAffine(tpl_binary, M_anchor, (fw, fh),
                                                    flags=cv2.INTER_NEAREST)
                    if tpl_contour is not None:
                        pts = tpl_contour.reshape(-1, 2).astype(np.float64)
                        ones = np.ones((pts.shape[0], 1))
                        pts_h = np.hstack([pts, ones])
                        wc_a = (M_anchor @ pts_h.T).T.reshape(-1, 1, 2).astype(np.int32)
                        cv2.drawContours(vis, [wc_a], -1, (255, 0, 255), 2)

                    complete_preview = np.maximum(sam_2d, warped_anchor)
                    comp_contours, _ = cv2.findContours(
                        complete_preview, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, comp_contours, -1, (255, 255, 0), 1)

                    iou_anchor = int(np.sum((sam_2d > 0) & (warped_anchor > 0))) / max(
                        int(np.sum((sam_2d > 0) | (warped_anchor > 0))), 1)

                    info_lines.append(f"\n━━━ 🔽 底部锚定对齐 (紫) ━━━")
                    info_lines.append(f"方法: {method_str}")
                    info_lines.append(f"中心: ({anchor_align['pos_x']:.0f},"
                                      f"{anchor_align['pos_y']:.0f})")
                    info_lines.append(f"角度: {anchor_align['angle_deg']:.1f}°")
                    info_lines.append(f"IoU: {iou_anchor:.1%}")
                    info_lines.append(f"完整mask像素: {int(np.sum(complete_preview > 0))}")

                    if sam_bottom:
                        db = np.sqrt((sam_bottom[0] - p_b_frame[0])**2 +
                                     (sam_bottom[1] - p_b_frame[1])**2)
                        info_lines.append(f"\nSAM底部↔Step1底部偏移: {db:.1f}px")
                        if db > 10:
                            info_lines.append("  ⚠️ 偏移较大,建议调整Step1对齐")

                    info_lines.append(f"\n🟢绿=Step1  🟣紫=底部锚定  🟡黄=完整轮廓")

            cv2.putText(vis, "Red=SAM Green=Step1 Purple=Anchor Cyan=EP",
                        (10, fh - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        else:
            info_lines.append("\n(Step 1 未设置对齐, 仅显示 SAM mask)")
            cv2.putText(vis, "Red=SAM  Green=BOTTOM  Cyan=TOP",
                        (10, fh - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        return vis_rgb, "\n".join(info_lines)

    except Exception as e:
        traceback.print_exc()
        return None, f"预览失败: {e}"

def on_s2_propagate_and_complete(
    video_dir_name, template_path, frame_idx,
    points_state, labels_state, class_id,
    pos_x, pos_y, scale, angle_deg, dilate_px,
    track_angle, use_endpoint_matching, progress=gr.Progress()
):

    global _propagation_results

    if not video_dir_name or not template_path:
        return "请先完成 Step 1", None
    if not points_state:
        return "请先标注至少一个点", None
    if pos_x is None or pos_y is None:
        return "请先在 Step 1 完成模板对齐 (位置未设定)", None

    pos_x, pos_y = float(pos_x), float(pos_y)
    frame_idx = int(frame_idx)
    class_id_int = int(class_id)
    dilate_px = int(dilate_px)

    try:
        log_lines = []
        def log(msg):
            log_lines.append(msg)
            print(msg)

        _, tpl_binary, tpl_contour, tpl_center = load_template(template_path)
        if tpl_binary is None:
            return "模板加载失败", None

        tpl_endpoints = get_template_endpoints(tpl_binary)

        video_path = os.path.join(BASE_DIR, video_dir_name)
        frames = get_sorted_frames(video_dir_name)
        sample_bgr = cv2.imread(os.path.join(video_path, frames[0]))
        fh, fw = sample_bgr.shape[:2]

        log(f"📂 视频: {video_dir_name}, {len(frames)} 帧, {fw}x{fh}")
        log(f"📐 模板: {os.path.basename(template_path)}")
        log(f"🎯 标注帧: {frame_idx}, 选点: {len(points_state)}")
        log(f"📍 Step1 对齐: pos=({pos_x:.0f},{pos_y:.0f}), "
            f"scale={scale:.4f}, angle={angle_deg:.1f}°")
        log(f"🔄 角度追踪: {'开启' if track_angle else '关闭'}")
        log(f"🔵 端点匹配: {'开启' if use_endpoint_matching else '关闭'}")
        log(f"📏 模板膨胀: {dilate_px}px")
        if tpl_endpoints:
            log(f"   模板端点: EP1=({tpl_endpoints['p1'][0]:.0f},{tpl_endpoints['p1'][1]:.0f}), "
                f"EP2=({tpl_endpoints['p2'][0]:.0f},{tpl_endpoints['p2'][1]:.0f}), "
                f"距离={tpl_endpoints['dist']:.0f}")
        else:
            log(f"   ⚠️ 模板端点检测失败, 将使用质心方法")
            use_endpoint_matching = False

        progress(0.0, desc="初始化 SAM2 ...")
        predictor = get_predictor()
        inference_state = predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

        points_np = np.array(points_state, dtype=np.float32)
        labels_np = np.array(labels_state, dtype=np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx, obj_id=1,
            points=points_np, labels=labels_np)
        log(f"✅ SAM 提示已添加")

        progress(0.1, desc="SAM 传播中 ...")
        video_segments = {}
        for out_fi, out_ids, out_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_fi] = {
                oid: (out_logits[i] > 0.0).cpu().numpy()
                for i, oid in enumerate(out_ids)
            }
        predictor.reset_state(inference_state)
        n_with_mask = sum(1 for segs in video_segments.values()
                         for m in segs.values() if np.any(np.squeeze(m) > 0))
        log(f"✅ SAM 传播完成: {n_with_mask}/{len(frames)} 帧有 mask")

        bottom_tpl_key = find_bottom_tpl_key(
            tpl_endpoints, tpl_center, pos_x, pos_y, scale, angle_deg)
        top_tpl_key = "p2" if bottom_tpl_key == "p1" else "p1"

        if tpl_endpoints is not None and use_endpoint_matching:
            M_ref = make_affine(tpl_center, pos_x, pos_y, scale, angle_deg)
            p_bottom = M_ref @ np.array([
                tpl_endpoints[bottom_tpl_key][0],
                tpl_endpoints[bottom_tpl_key][1], 1.0])
            p_top = M_ref @ np.array([
                tpl_endpoints[top_tpl_key][0],
                tpl_endpoints[top_tpl_key][1], 1.0])
            prev_bottom_pos = (float(p_bottom[0]), float(p_bottom[1]))
            prev_top_pos = (float(p_top[0]), float(p_top[1]))
            log(f"🔽 底部端点(模板{bottom_tpl_key}): "
                f"初始位置=({prev_bottom_pos[0]:.0f},{prev_bottom_pos[1]:.0f})")
            log(f"🔼 顶部端点(模板{top_tpl_key}): "
                f"初始位置=({prev_top_pos[0]:.0f},{prev_top_pos[1]:.0f})")
        else:
            prev_bottom_pos = None
            prev_top_pos = None

        last_good_angle = float(angle_deg)

        progress(0.3, desc="模板补全中 ...")
        completed_masks = {}
        warped_templates = {}
        pose_log = []
        method_counts = {}
        pose_log_methods = {}                             

        for fi in range(len(frames)):
            prog = 0.3 + 0.5 * fi / len(frames)
            progress(prog, desc=f"补全帧 {fi+1}/{len(frames)} ...")

            sam_mask_2d = np.zeros((fh, fw), dtype=np.uint8)
            has_sam = False
            if fi in video_segments:
                for oid, mask in video_segments[fi].items():
                    m2d = np.squeeze(mask).astype(np.uint8)
                    if m2d.shape[:2] != (fh, fw):
                        m2d = cv2.resize(m2d, (fw, fh), interpolation=cv2.INTER_NEAREST)
                    sam_mask_2d = (m2d > 0).astype(np.uint8) * 255
                    if np.any(sam_mask_2d > 0):
                        has_sam = True
                    break

            if has_sam:
                comp, warped, tracking = complete_single_mask(
                    sam_mask_2d, tpl_binary, tpl_center,
                    scale, angle_deg,
                    tpl_endpoints=tpl_endpoints,
                    use_endpoint_matching=use_endpoint_matching,
                    bottom_tpl_key=bottom_tpl_key,
                    prev_bottom_pos=prev_bottom_pos,
                    prev_top_pos=prev_top_pos,
                    last_good_angle=last_good_angle,
                    dilate_px=dilate_px)
                completed_masks[fi] = comp
                warped_templates[fi] = warped
                if tracking:
                    method = tracking["method"]
                    method_counts[method] = method_counts.get(method, 0) + 1
                    pose_log_methods[fi] = method

                    if tracking["bottom_pos"] is not None:
                        prev_bottom_pos = tracking["bottom_pos"]
                    if tracking["top_pos"] is not None:
                        prev_top_pos = tracking["top_pos"]

                    if track_angle and "dual_endpoint" in method:
                        last_good_angle = tracking["angle_deg"]
                    pose_log.append(
                        f"  帧{fi}: pos=({tracking['pos'][0]:.0f},{tracking['pos'][1]:.0f}), "
                        f"angle={tracking['angle_deg']:.1f}°, "
                        f"bottom=({tracking['bottom_pos'][0]:.0f},{tracking['bottom_pos'][1]:.0f})"
                        if tracking['bottom_pos'] else
                        f"  帧{fi}: 方法={method}(无底部端点)")
            else:
                completed_masks[fi] = sam_mask_2d
                warped_templates[fi] = np.zeros((fh, fw), dtype=np.uint8)

        n_completed = sum(1 for m in completed_masks.values() if np.any(m > 0))
        log(f"\n✅ 模板补全完成: {n_completed}/{len(frames)} 帧有完整 mask")
        log(f"   对齐方法统计:")
        for method, cnt in sorted(method_counts.items()):
            log(f"     {method}: {cnt} 帧")
        if pose_log:
            log(f"\n前几帧 pose (共 {len(pose_log)}):")
            for line in pose_log[:8]:
                log(line)
            if len(pose_log) > 8:
                log(f"  ... 等 {len(pose_log)} 帧")

        progress(0.8, desc="导出数据集 ...")
        output_dir = os.path.join(SAM2_DATASET_DIR, video_dir_name)
        images_dir = os.path.join(output_dir, 'images')
        labels_dir = os.path.join(output_dir, 'labels')
        vis_dir = os.path.join(output_dir, 'images_vis')
        vis_tpl_only_dir = os.path.join(output_dir, 'images_vis_template_only')
        for d in [images_dir, labels_dir, vis_dir, vis_tpl_only_dir]:
            os.makedirs(d, exist_ok=True)

        import csv as csv_mod
        csv_path = os.path.join(output_dir, 'tracking_results.csv')
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv_mod.writer(csv_file)
        csv_writer.writerow([
            'frame', 'bottom_x', 'bottom_y', 'top_x', 'top_y',
            'centroid_x', 'centroid_y', 'angle_deg', 'iou', 'method',
        ])

        saved = 0
        skipped = 0
        for fi in range(len(frames)):
            prog = 0.8 + 0.18 * fi / len(frames)
            progress(prog, desc=f"导出帧 {fi+1}/{len(frames)} ...")

            img_path = os.path.join(video_path, frames[fi])
            img = Image.open(img_path).convert('RGB')
            img_w, img_h = img.size
            img_name = f'{fi:05d}.jpg'
            shutil.copy(img_path, os.path.join(images_dir, img_name))

            mask = completed_masks.get(fi, np.zeros((fh, fw), dtype=np.uint8))
            warped_tpl = warped_templates.get(fi, np.zeros((fh, fw), dtype=np.uint8))

            tpl_ep_frame = find_endpoint_circles(warped_tpl) if np.any(warped_tpl > 0) else None

            tpl_pose = get_mask_pose(warped_tpl) if np.any(warped_tpl > 0) else None

            sam_m = np.zeros((fh, fw), dtype=np.uint8)
            if fi in video_segments:
                for oid, m in video_segments[fi].items():
                    m2d = np.squeeze(m).astype(np.uint8)
                    if m2d.shape[:2] != (fh, fw):
                        m2d = cv2.resize(m2d, (fw, fh), interpolation=cv2.INTER_NEAREST)
                    sam_m = (m2d > 0).astype(np.uint8) * 255
                    break

            sam_bool = sam_m > 0
            tpl_bool = warped_tpl > 0
            intersection = int(np.sum(sam_bool & tpl_bool))
            union = int(np.sum(sam_bool | tpl_bool))
            iou = intersection / max(union, 1)

            if tpl_ep_frame is not None and tpl_pose is not None:

                ep1 = np.array(tpl_ep_frame["p1"])
                ep2 = np.array(tpl_ep_frame["p2"])
                if ep1[1] >= ep2[1]:
                    bot_pos, top_pos = ep1, ep2
                else:
                    bot_pos, top_pos = ep2, ep1
                csv_writer.writerow([
                    fi,
                    f'{bot_pos[0]:.2f}', f'{bot_pos[1]:.2f}',
                    f'{top_pos[0]:.2f}', f'{top_pos[1]:.2f}',
                    f'{tpl_pose["cx"]:.2f}', f'{tpl_pose["cy"]:.2f}',
                    f'{tpl_pose["angle_deg"]:.2f}',
                    f'{iou:.4f}',
                    pose_log_methods.get(fi, 'unknown'),
                ])
            else:
                csv_writer.writerow([
                    fi, 0, 0, 0, 0, 0, 0, 0, f'{iou:.4f}', 'no_template',
                ])

            label_lines = []
            if np.any(mask > 0):
                polygon = mask_to_yolo_seg(mask, img_w, img_h)
                if polygon and len(polygon) >= 6:
                    poly_arr = np.array(polygon)
                    if not (np.any(poly_arr < 0) or np.any(poly_arr > 1)):
                        poly_str = ' '.join(f'{c:.6f}' for c in polygon)
                        label_lines.append(f"{class_id_int} {poly_str}\n")

            with open(os.path.join(labels_dir, f'{fi:05d}.txt'), 'w') as f:
                f.writelines(label_lines)
            if not label_lines:
                skipped += 1

            frame_bgr = cv2.imread(img_path)
            vis_rgb = build_completion_vis(
                frame_bgr,
                video_segments.get(fi, {}).get(1, np.zeros((fh, fw), dtype=np.uint8)),
                mask, warped_tpl)
            cv2.imwrite(os.path.join(vis_dir, img_name),
                        cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))

            vis_tpl_only_rgb = build_template_only_vis(frame_bgr, warped_tpl)
            cv2.imwrite(os.path.join(vis_tpl_only_dir, img_name),
                        cv2.cvtColor(vis_tpl_only_rgb, cv2.COLOR_RGB2BGR))
            saved += 1

        csv_file.close()

        progress(1.0, desc="完成!")
        log(f"\n{'='*50}")
        log(f"   导出完成!")
        log(f"   总帧数: {len(frames)}")
        log(f"   有 mask 的帧: {saved - skipped}")
        log(f"   无 mask 的帧: {skipped}")
        log(f"   输出目录: {output_dir}")
        log(f"     ├── images/     (原图)")
        log(f"     ├── labels/     (YOLO 标注)")
        log(f"     ├── images_vis/ (可视化1: 蓝=SAM, 黄=模板补全, 绿=最终轮廓)")
        log(f"     ├── images_vis_template_only/ (可视化2: 仅补全mask, 无SAM)")
        log(f"     └── tracking_results.csv (端点质心+目标质心+IoU)")
        log(f"\nCSV 列说明:")
        log(f"  frame        - 帧序号")
        log(f"  bottom_x/y   - 底部端点圆质心 (px)")
        log(f"  top_x/y      - 顶部端点圆质心 (px)")
        log(f"  centroid_x/y - 整个目标(模板)质心 (px)")
        log(f"  angle_deg    - 模板旋转角 (度)")
        log(f"  iou          - SAM分割 vs 匹配mask 的 IoU")
        log(f"  method       - 对齐方法")

        _propagation_results["video_dir"] = video_dir_name
        _propagation_results["completed_masks"] = completed_masks
        _propagation_results["warped_templates"] = warped_templates
        _propagation_results["video_segments"] = video_segments
        _propagation_results["frames"] = frames

        first_vis_path = os.path.join(vis_dir, '00000.jpg')
        first_vis = None
        if os.path.exists(first_vis_path):
            first_vis = cv2.cvtColor(cv2.imread(first_vis_path), cv2.COLOR_BGR2RGB)

        return "\n".join(log_lines), first_vis

    except Exception as e:
        traceback.print_exc()
        return f"❌ 失败: {e}\n{traceback.format_exc()}", None

def on_s2_browse(video_dir_name, frame_idx):

    if not _propagation_results or _propagation_results.get("video_dir") != video_dir_name:
        return None, "请先运行 SAM传播+模板补全"
    try:
        frames = _propagation_results["frames"]
        idx = max(0, min(int(frame_idx), len(frames) - 1))
        completed = _propagation_results["completed_masks"]
        warped_tpls = _propagation_results["warped_templates"]
        video_segments = _propagation_results["video_segments"]

        frame_bgr = cv2.imread(os.path.join(BASE_DIR, video_dir_name, frames[idx]))
        fh, fw = frame_bgr.shape[:2]

        sam_mask = np.zeros((fh, fw), dtype=np.uint8)
        if idx in video_segments:
            for oid, m in video_segments[idx].items():
                m2d = np.squeeze(m).astype(np.uint8)
                if m2d.shape[:2] != (fh, fw):
                    m2d = cv2.resize(m2d, (fw, fh), interpolation=cv2.INTER_NEAREST)
                sam_mask = (m2d > 0).astype(np.uint8) * 255
                break

        comp = completed.get(idx, np.zeros((fh, fw), dtype=np.uint8))
        wt = warped_tpls.get(idx, np.zeros((fh, fw), dtype=np.uint8))

        vis = build_completion_vis(frame_bgr, sam_mask, comp, wt)

        sam_area = int(np.sum(sam_mask > 0))
        comp_area = int(np.sum(comp > 0))
        added = comp_area - sam_area
        info = (f"帧 {idx}/{len(frames)-1} | "
                f"SAM: {sam_area}px² → 补全后: {comp_area}px² "
                f"(+{added}px², +{added/max(sam_area,1)*100:.0f}%)")
        return vis, info
    except Exception as e:
        return None, f"错误: {e}"

def build_app():
    video_dirs = list_video_dirs()

    with gr.Blocks(title="模板对齐补全工具") as app:
        gr.Markdown("# 🔧 2D模板对齐 + SAM补全 一站式工具")
        gr.Markdown(
            "**Step 1**: 手动对齐模板到第一帧 (点击定位 + 缩放/旋转)\n\n"
            "**Step 2**: SAM 传播 + 每帧根据SAM mask动态追踪位姿 → 模板补全 → 导出 YOLO"
        )

        with gr.Accordion("Step 1: 模板对齐 (手动)", open=True):
            with gr.Row():
                video_dir_dd = gr.Dropdown(
                    choices=video_dirs, label="选择视频帧目录",
                    info="video_to_img/ 下的子文件夹")
                template_path_input = gr.Textbox(
                    value=DEFAULT_TEMPLATE, label="模板图片路径")
                load_btn = gr.Button("加载数据", variant="primary")

            pos_x_state = gr.State(None)
            pos_y_state = gr.State(None)

            with gr.Row():
                with gr.Column(scale=1):
                    tpl_img = gr.Image(label="模板轮廓", interactive=False, height=220)
                    gr.Markdown("📍 **点击右图定位** | 🎚 滑条调缩放/旋转")
                    scale_slider = gr.Slider(0.01, 0.5, 0.075, step=0.005, label="缩放")
                    angle_slider = gr.Slider(-180, 180, -92.5, step=0.5, label="旋转 (度)")
                    dilate_slider = gr.Slider(0, 10, 3, step=1, label="模板膨胀 (px)",
                                              info="加粗中间细杆,0=不膨胀")
                    alpha_slider = gr.Slider(10, 80, 40, step=5, label="透明度 %")
                    with gr.Row():
                        update_btn = gr.Button("刷新", variant="secondary")
                        save_btn = gr.Button("💾 保存对齐", variant="primary")
                    info_box = gr.Textbox(label="状态", lines=3, interactive=False)
                with gr.Column(scale=2):
                    preview_img = gr.Image(
                        label="点击设置位置 | 绿线=轮廓 | 青色=区域 | 红十字=中心",
                        interactive=False)
                    preview_info = gr.Textbox(label="预览", lines=1, interactive=False)
            save_result = gr.Textbox(label="保存结果", lines=3, interactive=False)

        with gr.Accordion("Step 2: SAM传播 + 模板补全 + YOLO导出", open=True):
            gr.Markdown(
                "在下方标注目标 → 运行传播+补全\n\n"
                "🔵 蓝色 = SAM原始mask | 🟡 黄色 = 模板补全部分 | 🟢 绿色轮廓 = 最终完整mask"
            )
            s2_pts_state = gr.State([])
            s2_labels_state = gr.State([])

            with gr.Row():
                with gr.Column(scale=1):
                    s2_frame_slider = gr.Slider(0, 1000, 0, step=1, label="标注帧索引")
                    s2_point_type = gr.Radio(
                        ["正样本 (前景)", "负样本 (背景)"],
                        value="正样本 (前景)", label="点击类型")
                    s2_class_id = gr.Number(value=0, label="CLASS_ID", precision=0)
                    s2_track_angle = gr.Checkbox(value=True, label="角度追踪",
                                                 info="从SAM mask追踪旋转角度变化")
                    s2_use_endpoints = gr.Checkbox(value=True, label="🔵 端点圆匹配",
                                                   info="用两端圆形特征定位(推荐),否则用质心")
                    with gr.Row():
                        s2_clear_btn = gr.Button("清除标注", variant="secondary")
                        s2_preview_btn = gr.Button("👁 预览 SAM Mask",
                                                   variant="secondary")
                    s2_run_btn = gr.Button("🚀 传播 + 补全 + 导出 YOLO",
                                           variant="primary", size="lg")
                    s2_pts_info = gr.Textbox(label="标注", lines=2, interactive=False)

                with gr.Column(scale=2):
                    s2_annotate_img = gr.Image(
                        label="点击标注目标 (绿=前景, 红=背景)",
                        interactive=False)

            gr.Markdown("#### 👁 SAM Mask 预览 (对比模板轮廓)")
            gr.Markdown(
                "🔴 红色轮廓/蓝色填充 = SAM分割结果 | 🟢 绿色轮廓 = 模板 | "
                "蓝十字 = SAM质心 | 红十字 = 模板中心"
            )
            with gr.Row():
                s2_preview_img = gr.Image(label="SAM mask vs 模板对比", interactive=False)
                s2_preview_info = gr.Textbox(label="SAM vs 模板 对比信息", lines=12,
                                             interactive=False)

            s2_log = gr.Textbox(label="运行日志", lines=15, interactive=False)

            gr.Markdown("#### 浏览补全结果")
            with gr.Row():
                s2_browse_slider = gr.Slider(0, 1000, 0, step=1, label="帧索引")
                s2_browse_btn = gr.Button("查看", variant="secondary")
            s2_browse_img = gr.Image(label="补全可视化", interactive=False)
            s2_browse_info = gr.Textbox(label="帧信息", lines=1, interactive=False)

        load_btn.click(
            fn=on_load_data,
            inputs=[video_dir_dd, template_path_input],
            outputs=[tpl_img, preview_img, pos_x_state, pos_y_state,
                     scale_slider, info_box])

        preview_img.select(
            fn=on_click_frame,
            inputs=[pos_x_state, pos_y_state, video_dir_dd, template_path_input,
                    scale_slider, angle_slider, dilate_slider, alpha_slider],
            outputs=[preview_img, pos_x_state, pos_y_state, preview_info])

        _s_in = [pos_x_state, pos_y_state, video_dir_dd, template_path_input,
                 scale_slider, angle_slider, dilate_slider, alpha_slider]
        _s_out = [preview_img, preview_info]
        for ctrl in [scale_slider, angle_slider, dilate_slider, alpha_slider]:
            ctrl.release(fn=on_slider_change, inputs=_s_in, outputs=_s_out)
        update_btn.click(fn=on_slider_change, inputs=_s_in, outputs=_s_out)

        save_btn.click(
            fn=on_save_params,
            inputs=[pos_x_state, pos_y_state, video_dir_dd,
                    template_path_input, scale_slider, angle_slider],
            outputs=[save_result])

        def s2_on_dir_or_frame(vdir, fi):
            img, info = on_s2_load_frame(vdir, fi)
            frames = get_sorted_frames(vdir) if vdir else []
            return (img, info,
                    gr.update(maximum=max(len(frames)-1, 0)),
                    gr.update(maximum=max(len(frames)-1, 0)),
                    [], [])

        video_dir_dd.change(
            fn=s2_on_dir_or_frame,
            inputs=[video_dir_dd, s2_frame_slider],
            outputs=[s2_annotate_img, s2_pts_info,
                     s2_frame_slider, s2_browse_slider,
                     s2_pts_state, s2_labels_state])

        s2_frame_slider.release(
            fn=lambda vd, fi: on_s2_load_frame(vd, fi),
            inputs=[video_dir_dd, s2_frame_slider],
            outputs=[s2_annotate_img, s2_pts_info])

        s2_annotate_img.select(
            fn=on_s2_click,
            inputs=[video_dir_dd, s2_frame_slider, s2_point_type,
                    s2_pts_state, s2_labels_state],
            outputs=[s2_annotate_img, s2_pts_state, s2_labels_state, s2_pts_info])

        s2_clear_btn.click(
            fn=on_s2_clear,
            inputs=[video_dir_dd, s2_frame_slider],
            outputs=[s2_annotate_img, s2_pts_state, s2_labels_state, s2_pts_info])

        s2_preview_btn.click(
            fn=on_s2_preview_mask,
            inputs=[video_dir_dd, template_path_input, s2_frame_slider,
                    s2_pts_state, s2_labels_state,
                    pos_x_state, pos_y_state, scale_slider, angle_slider],
            outputs=[s2_preview_img, s2_preview_info])

        s2_run_btn.click(
            fn=on_s2_propagate_and_complete,
            inputs=[video_dir_dd, template_path_input, s2_frame_slider,
                    s2_pts_state, s2_labels_state, s2_class_id,
                    pos_x_state, pos_y_state, scale_slider, angle_slider,
                    dilate_slider,
                    s2_track_angle, s2_use_endpoints],
            outputs=[s2_log, s2_browse_img])

        s2_browse_btn.click(
            fn=on_s2_browse,
            inputs=[video_dir_dd, s2_browse_slider],
            outputs=[s2_browse_img, s2_browse_info])

    return app

if __name__ == '__main__':
    app = build_app()
    app.launch(
        server_name='0.0.0.0',
        server_port=7862,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
