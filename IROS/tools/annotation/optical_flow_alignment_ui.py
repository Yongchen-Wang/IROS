#!/usr/bin/env python3

import os
import sys
import shutil
import numpy as np
import cv2
from PIL import Image, ImageDraw
import gradio as gr
import traceback
from pathlib import Path
from iros.paths import ASSET_ROOT, FRAME_ROOT, OUTPUT_ROOT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = str(FRAME_ROOT)
DEFAULT_TEMPLATE = str(ASSET_ROOT / "robot_template.jpg")
OUTPUT_BASE = str(OUTPUT_ROOT / "mask_alignment")

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
FB_THRESHOLD = 5.0                        
ANGLE_JUMP_LIMIT = 30.0              
DIST_TOLERANCE = 0.15                                  

def list_video_dirs():

    if not os.path.isdir(BASE_DIR):
        return []
    dirs = []
    for d in sorted(os.listdir(BASE_DIR)):
        full = os.path.join(BASE_DIR, d)
        if os.path.isdir(full):
            if any(f.lower().endswith(('.jpg', '.jpeg')) for f in os.listdir(full)):
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
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_connected = cv2.dilate(bin_inv, k, iterations=2)
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

def make_affine(tpl_center, pos_x, pos_y, scale, angle_deg):

    tcx, tcy = tpl_center
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [scale * c, -scale * s, pos_x - scale * (c * tcx - s * tcy)],
        [scale * s,  scale * c, pos_y - scale * (s * tcx + c * tcy)],
    ], dtype=np.float64)

def warp_template(tpl_binary, tpl_center, pos_x, pos_y, scale, angle_deg,
                  fw, fh, dilate_px=0):

    M = make_affine(tpl_center, pos_x, pos_y, scale, angle_deg)
    warped = cv2.warpAffine(tpl_binary, M, (fw, fh), flags=cv2.INTER_NEAREST)
    if dilate_px > 0:
        ks = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        warped = cv2.dilate(warped, kernel, iterations=1)
    return warped

def find_bottom_tpl_key(tpl_ep, tpl_center, pos_x, pos_y, scale, angle_deg):

    if tpl_ep is None:
        return "p1"
    M = make_affine(tpl_center, pos_x, pos_y, scale, angle_deg)
    p1_f = M @ np.array([tpl_ep["p1"][0], tpl_ep["p1"][1], 1.0])
    p2_f = M @ np.array([tpl_ep["p2"][0], tpl_ep["p2"][1], 1.0])
    return "p1" if p1_f[1] >= p2_f[1] else "p2"

def align_from_anchor(tpl_ep, tpl_center, anchor_key, anchor_pos, scale, angle_deg):

    ta = tpl_ep[anchor_key]
    dx = tpl_center[0] - ta[0]
    dy = tpl_center[1] - ta[1]
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    return {
        "pos_x": float(anchor_pos[0] + scale * (c * dx - s * dy)),
        "pos_y": float(anchor_pos[1] + scale * (s * dx + c * dy)),
        "scale": scale,
        "angle_deg": angle_deg,
    }

def compute_angle_from_endpoints(bottom_pos, top_pos, tpl_ep, bottom_tpl_key):

    top_key = "p2" if bottom_tpl_key == "p1" else "p1"
    dx_f = top_pos[0] - bottom_pos[0]
    dy_f = top_pos[1] - bottom_pos[1]
    tb = tpl_ep[bottom_tpl_key]
    tt = tpl_ep[top_key]
    dx_t = tt[0] - tb[0]
    dy_t = tt[1] - tb[1]
    return np.degrees(np.arctan2(dy_f, dx_f)) - np.degrees(np.arctan2(dy_t, dx_t))

def mask_to_yolo_seg(mask, img_w, img_h, simplify_tolerance=2.0):

    m2d = np.squeeze(mask)
    if m2d.ndim != 2:
        return None
    mu = (m2d > 0).astype(np.uint8) * 255
    mh, mw = m2d.shape
    if mh != img_h or mw != img_w:
        mu = cv2.resize(mu, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(mu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if simplify_tolerance > 0:
        c = cv2.approxPolyDP(c, simplify_tolerance * cv2.arcLength(c, True) / 100, True)
    if len(c) < 3:
        return None
    poly = c.reshape(-1, 2).astype(np.float32)
    poly[:, 0] /= img_w
    poly[:, 1] /= img_h
    return poly.flatten().tolist()

def build_overlay(frame_bgr, tpl_binary, tpl_contour, tpl_center,
                  pos_x, pos_y, scale, angle_deg, alpha, dilate_px=0):

    fh, fw = frame_bgr.shape[:2]
    M = make_affine(tpl_center, pos_x, pos_y, scale, angle_deg)
    wm = cv2.warpAffine(tpl_binary, M, (fw, fh), flags=cv2.INTER_NEAREST)
    if dilate_px > 0:
        ks = 2 * dilate_px + 1
        wm = cv2.dilate(wm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)),
                         iterations=1)

    wc = None
    if tpl_contour is not None:
        pts = tpl_contour.reshape(-1, 2).astype(np.float64)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        wc = (M @ pts_h.T).T.reshape(-1, 1, 2).astype(np.int32)

    vis = frame_bgr.copy()
    ov = np.zeros_like(vis)
    ov[:] = (0, 200, 200)
    m3 = (wm > 0).astype(np.float32)[..., None]
    vis = (vis * (1 - m3 * alpha) + ov * m3 * alpha).astype(np.uint8)
    if wc is not None:
        cv2.drawContours(vis, [wc], -1, (0, 255, 0), 2)
    cv2.drawMarker(vis, (int(pos_x), int(pos_y)), (0, 0, 255),
                   cv2.MARKER_CROSS, 20, 2)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis_rgb, wm, int(np.sum(wm > 0))

def build_vis_with_markers(frame_bgr, warped, bottom_pos, top_pos,
                           centroid, method=""):

    vis = frame_bgr.copy()

    m3 = (warped > 0).astype(np.float32)[..., None]
    ov = np.zeros_like(vis)
    ov[:] = (0, 200, 200)              
    vis = (vis * (1 - m3 * 0.45) + ov * m3 * 0.45).astype(np.uint8)

    cnts, _ = cv2.findContours((warped > 0).astype(np.uint8) * 255,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)

    bx, by = int(bottom_pos[0]), int(bottom_pos[1])
    cv2.circle(vis, (bx, by), 10, (0, 255, 0), 2)
    cv2.circle(vis, (bx, by), 3, (0, 255, 0), -1)
    cv2.putText(vis, "B", (bx + 12, by - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    tx, ty = int(top_pos[0]), int(top_pos[1])
    cv2.circle(vis, (tx, ty), 10, (255, 255, 0), 2)
    cv2.circle(vis, (tx, ty), 3, (255, 255, 0), -1)
    cv2.putText(vis, "T", (tx + 12, ty - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    cv2.line(vis, (bx, by), (tx, ty), (200, 200, 200), 1, cv2.LINE_AA)

    cx_i, cy_i = int(centroid[0]), int(centroid[1])
    cv2.drawMarker(vis, (cx_i, cy_i), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
    cv2.putText(vis, "C", (cx_i + 12, cy_i - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    if method:
        cv2.putText(vis, method, (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

def build_template_only_vis(frame_bgr, warped):

    vis = frame_bgr.copy()
    m3 = (warped > 0).astype(np.float32)[..., None]
    ov = np.zeros_like(vis)
    ov[:] = (0, 200, 200)
    vis = (vis * (1 - m3 * 0.45) + ov * m3 * 0.45).astype(np.uint8)
    cnts, _ = cv2.findContours((warped > 0).astype(np.uint8) * 255,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

_cache = {}

def _get_cached(video_dir_name, template_path, frame_idx=0):

    key = (video_dir_name, template_path, frame_idx)
    if key not in _cache:
        bgr, binary, cnt, center = load_template(template_path)
        ep = get_template_endpoints(binary) if binary is not None else None
        frames = get_sorted_frames(video_dir_name)
        idx = max(0, min(frame_idx, len(frames) - 1))
        fbgr = cv2.imread(os.path.join(BASE_DIR, video_dir_name, frames[idx]))
        _cache[key] = dict(
            tpl_bgr=bgr, tpl_binary=binary, tpl_contour=cnt,
            tpl_center=center, tpl_endpoints=ep,
            frame_bgr=fbgr, frames=frames,
        )
    return _cache[key]

def on_load_data(video_dir_name, template_path, start_frame):

    global _cache
    _cache.clear()
    if not video_dir_name:
        return None, None, None, None, gr.update(), "请先选择视频目录"
    if not template_path or not os.path.exists(template_path):
        return None, None, None, None, gr.update(), f"模板不存在: {template_path}"
    try:
        start_frame = int(start_frame)
        d = _get_cached(video_dir_name, template_path, start_frame)
        bgr = d["tpl_bgr"]
        binary = d["tpl_binary"]
        cnt = d["tpl_contour"]
        center = d["tpl_center"]
        fbgr = d["frame_bgr"]
        frames = d["frames"]
        if bgr is None:
            return None, None, None, None, gr.update(), "模板读取失败"

        fh, fw = fbgr.shape[:2]
        th, tw = bgr.shape[:2]

        tpl_vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
        if cnt is not None:
            cv2.drawContours(tpl_vis, [cnt], -1, (0, 255, 0), 4)
            cv2.drawMarker(tpl_vis, (int(center[0]), int(center[1])),
                           (255, 0, 0), cv2.MARKER_CROSS, 30, 3)

        ix, iy = float(fw / 2), float(fh / 2)
        vis, _, _ = build_overlay(fbgr, binary, cnt, center,
                                  ix, iy, 0.07, -92.5, 0.4)
        info = (f"✅ 视频: {video_dir_name} ({len(frames)}帧, {fw}x{fh}), "
                f"模板: {tw}x{th}, 起始帧: {start_frame}\n"
                f"👆 点击右图定位 → 滑条调缩放/旋转/膨胀")
        return tpl_vis, vis, ix, iy, gr.update(value=0.07), info
    except Exception as e:
        traceback.print_exc()
        return None, None, None, None, gr.update(), f"加载失败: {e}"

def on_click_frame(pos_x, pos_y, video_dir_name, template_path, start_frame,
                   scale, angle_deg, dilate_px, alpha, evt: gr.SelectData):

    x, y = float(evt.index[0]), float(evt.index[1])
    try:
        d = _get_cached(video_dir_name, template_path, int(start_frame))
        vis, _, area = build_overlay(
            d["frame_bgr"], d["tpl_binary"], d["tpl_contour"],
            d["tpl_center"], x, y, scale, angle_deg, alpha / 100.0,
            dilate_px=int(dilate_px))
        info = (f"📍 ({x:.0f}, {y:.0f}), scale={scale:.4f}, "
                f"angle={angle_deg:.1f}°, dilate={int(dilate_px)}px, "
                f"area={area}px²")
        return vis, x, y, info
    except Exception as e:
        return None, pos_x, pos_y, f"错误: {e}"

def on_slider_change(pos_x, pos_y, video_dir_name, template_path, start_frame,
                     scale, angle_deg, dilate_px, alpha):

    if not video_dir_name or not template_path:
        return None, "请先加载数据"
    if pos_x is None or pos_y is None:
        return None, "请先点击预览图设置位置"
    try:
        d = _get_cached(video_dir_name, template_path, int(start_frame))
        vis, _, area = build_overlay(
            d["frame_bgr"], d["tpl_binary"], d["tpl_contour"],
            d["tpl_center"], float(pos_x), float(pos_y),
            scale, angle_deg, alpha / 100.0, dilate_px=int(dilate_px))
        info = (f"📍 ({pos_x:.0f}, {pos_y:.0f}), scale={scale:.4f}, "
                f"angle={angle_deg:.1f}°, dilate={int(dilate_px)}px, "
                f"area={area}px²")
        return vis, info
    except Exception as e:
        return None, f"错误: {e}"

_tracking_results = {}

def _rigid_correct(anchor, other, init_dist, last_angle_rad):

    ox = anchor[0] + init_dist * np.cos(last_angle_rad)
    oy = anchor[1] + init_dist * np.sin(last_angle_rad)
    return (float(ox), float(oy))

def _track_direction(video_path, frames, start_fi, direction,
                     init_pts, tpl_endpoints, bottom_key, init_angle,
                     track_angle, log_fn):

    results = {}
    pts = init_pts.copy()
    prev_gray = cv2.cvtColor(
        cv2.imread(os.path.join(video_path, frames[start_fi])),
        cv2.COLOR_BGR2GRAY)
    last_good_angle = init_angle

    init_dist = float(np.sqrt(
        (init_pts[1, 0, 0] - init_pts[0, 0, 0]) ** 2 +
        (init_pts[1, 0, 1] - init_pts[0, 0, 1]) ** 2))
    corrected_count = 0

    if direction > 0:
        frame_range = range(start_fi + 1, len(frames))
    else:
        frame_range = range(start_fi - 1, -1, -1)

    for fi in frame_range:
        curr_gray = cv2.cvtColor(
            cv2.imread(os.path.join(video_path, frames[fi])),
            cv2.COLOR_BGR2GRAY)

        new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None, **LK_PARAMS)

        back_pts, bst, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, new_pts, None, **LK_PARAMS)
        fb_err = np.sqrt(np.sum((pts - back_pts) ** 2, axis=2)).flatten()
        good = (st.flatten() == 1) & (fb_err < FB_THRESHOLD)

        adj_fi = fi - direction                        
        prev_r = results.get(adj_fi)
        if prev_r is None:
            prev_bottom = (float(init_pts[0, 0, 0]), float(init_pts[0, 0, 1]))
            prev_top = (float(init_pts[1, 0, 0]), float(init_pts[1, 0, 1]))
        else:
            prev_bottom = prev_r["bottom_pos"]
            prev_top = prev_r["top_pos"]

        if good[0] and good[1]:
            bot = (float(new_pts[0, 0, 0]), float(new_pts[0, 0, 1]))
            top = (float(new_pts[1, 0, 0]), float(new_pts[1, 0, 1]))
            method = "dual_lk"

            cur_dist = np.sqrt((top[0]-bot[0])**2 + (top[1]-bot[1])**2)
            ratio = cur_dist / init_dist if init_dist > 0 else 1.0
            if abs(ratio - 1.0) > DIST_TOLERANCE:

                angle_rad = np.deg2rad(last_good_angle)

                raw_rad = np.arctan2(top[1]-bot[1], top[0]-bot[0])
                top = _rigid_correct(bot, top, init_dist, raw_rad)
                method = "dual_lk(dist_corrected)"
                corrected_count += 1

            if track_angle:
                a = compute_angle_from_endpoints(bot, top, tpl_endpoints, bottom_key)
                if abs(((a - last_good_angle) + 180) % 360 - 180) < ANGLE_JUMP_LIMIT:
                    last_good_angle = a
                else:
                    method = method.replace(")", "+angle_clamped)") if ")" in method \
                             else method + "(angle_clamped)"

            results[fi] = {"bottom_pos": bot, "top_pos": top,
                           "angle_deg": last_good_angle, "method": method}

            pts[0, 0, 0] = bot[0]; pts[0, 0, 1] = bot[1]
            pts[1, 0, 0] = top[0]; pts[1, 0, 1] = top[1]

        elif good[0]:
            bot = (float(new_pts[0, 0, 0]), float(new_pts[0, 0, 1]))

            angle_rad = np.arctan2(
                prev_top[1] - prev_bottom[1],
                prev_top[0] - prev_bottom[0])
            top = _rigid_correct(bot, prev_top, init_dist, angle_rad)
            results[fi] = {"bottom_pos": bot, "top_pos": top,
                           "angle_deg": last_good_angle,
                           "method": "bottom_lk(rigid)"}
            pts[0] = new_pts[0]
            pts[1, 0, 0] = top[0]; pts[1, 0, 1] = top[1]

        elif good[1]:
            top = (float(new_pts[1, 0, 0]), float(new_pts[1, 0, 1]))

            angle_rad = np.arctan2(
                prev_top[1] - prev_bottom[1],
                prev_top[0] - prev_bottom[0])

            bot = (float(top[0] - init_dist * np.cos(angle_rad)),
                   float(top[1] - init_dist * np.sin(angle_rad)))
            results[fi] = {"bottom_pos": bot, "top_pos": top,
                           "angle_deg": last_good_angle,
                           "method": "top_lk(rigid)"}
            pts[0, 0, 0] = bot[0]; pts[0, 0, 1] = bot[1]
            pts[1] = new_pts[1]

        else:
            results[fi] = {"bottom_pos": prev_bottom, "top_pos": prev_top,
                           "angle_deg": last_good_angle, "method": "lost"}

        prev_gray = curr_gray

    if corrected_count > 0:
        log_fn(f"  {'⏩' if direction > 0 else '⏪'} "
               f"刚体距离修正: {corrected_count} 帧 "
               f"(容忍度 ±{DIST_TOLERANCE*100:.0f}%)")

    return results

def on_run_tracking(video_dir_name, template_path, start_frame,
                    pos_x, pos_y, scale, angle_deg, dilate_px,
                    class_id, track_angle, progress=gr.Progress()):

    global _tracking_results

    if not video_dir_name or not template_path:
        return "请先完成 Step 1", None
    if pos_x is None or pos_y is None:
        return "请先在 Step 1 设置位置 (点击预览图)", None

    pos_x, pos_y = float(pos_x), float(pos_y)
    start_fi = int(start_frame)
    class_id = int(class_id)
    dilate_px = int(dilate_px)

    try:
        log_lines = []

        def log(msg):
            log_lines.append(msg)
            print(msg)

        d = _get_cached(video_dir_name, template_path, start_fi)
        tpl_binary = d["tpl_binary"]
        tpl_contour = d["tpl_contour"]
        tpl_center = d["tpl_center"]
        tpl_endpoints = d["tpl_endpoints"]
        frames = d["frames"]
        video_path = os.path.join(BASE_DIR, video_dir_name)

        if tpl_binary is None:
            return "模板加载失败", None
        if tpl_endpoints is None:
            return "模板端点检测失败 (需要两端有圆形特征)", None

        sample_bgr = cv2.imread(os.path.join(video_path, frames[0]))
        fh, fw = sample_bgr.shape[:2]

        bottom_key = find_bottom_tpl_key(
            tpl_endpoints, tpl_center, pos_x, pos_y, scale, angle_deg)
        top_key = "p2" if bottom_key == "p1" else "p1"

        M_ref = make_affine(tpl_center, pos_x, pos_y, scale, angle_deg)
        p_bot = M_ref @ np.array([tpl_endpoints[bottom_key][0],
                                   tpl_endpoints[bottom_key][1], 1.0])
        p_top = M_ref @ np.array([tpl_endpoints[top_key][0],
                                   tpl_endpoints[top_key][1], 1.0])
        init_bottom = (float(p_bot[0]), float(p_bot[1]))
        init_top = (float(p_top[0]), float(p_top[1]))

        log(f"📂 视频: {video_dir_name}, {len(frames)} 帧, {fw}x{fh}")
        log(f"📐 模板: {os.path.basename(template_path)}")
        log(f"🎬 起始帧: {start_fi}")
        log(f"📍 Step1: pos=({pos_x:.0f},{pos_y:.0f}), "
            f"scale={scale:.4f}, angle={angle_deg:.1f}°")
        log(f"🔽 底部端点 (模板{bottom_key}): "
            f"({init_bottom[0]:.0f},{init_bottom[1]:.0f})")
        log(f"🔼 顶部端点 (模板{top_key}): "
            f"({init_top[0]:.0f},{init_top[1]:.0f})")
        log(f"📏 膨胀: {dilate_px}px")
        log(f"🔄 角度追踪: {'开启' if track_angle else '关闭'}")
        log(f"⚡ 方法: LK 稀疏光流 (FB阈值={FB_THRESHOLD}px)")

        progress(0.05, desc="光流追踪中...")

        init_pts = np.array([
            [init_bottom[0], init_bottom[1]],
            [init_top[0], init_top[1]],
        ], dtype=np.float32).reshape(-1, 1, 2)

        all_results = {
            start_fi: {
                "bottom_pos": init_bottom,
                "top_pos": init_top,
                "angle_deg": float(angle_deg),
                "method": "init",
            }
        }

        if start_fi < len(frames) - 1:
            log(f"\n⏩ 前向追踪: 帧 {start_fi} → {len(frames)-1} "
                f"({len(frames)-1-start_fi} 帧)")
            fwd = _track_direction(
                video_path, frames, start_fi, +1,
                init_pts, tpl_endpoints, bottom_key,
                float(angle_deg), track_angle, log)
            all_results.update(fwd)

        if start_fi > 0:
            log(f"⏪ 后向追踪: 帧 {start_fi} → 0 ({start_fi} 帧)")
            bwd = _track_direction(
                video_path, frames, start_fi, -1,
                init_pts, tpl_endpoints, bottom_key,
                float(angle_deg), track_angle, log)
            all_results.update(bwd)

        method_counts = {}
        for r in all_results.values():
            m = r["method"]
            method_counts[m] = method_counts.get(m, 0) + 1
        log(f"\n✅ 追踪完成! 共 {len(all_results)}/{len(frames)} 帧")
        log(f"方法统计:")
        for m, cnt in sorted(method_counts.items()):
            log(f"  {m}: {cnt} 帧")

        progress(0.4, desc="导出中...")
        out_dir = os.path.join(OUTPUT_BASE, video_dir_name)
        sub_dirs = ['images', 'labels', 'images_vis',
                    'images_vis_template_only', 'keypoints']
        for sub in sub_dirs:
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

        saved = 0
        skipped = 0
        for fi in range(len(frames)):
            progress(0.4 + 0.55 * fi / len(frames),
                     desc=f"导出帧 {fi+1}/{len(frames)}")

            frame_path = os.path.join(video_path, frames[fi])
            frame_bgr = cv2.imread(frame_path)
            img_name = f"{fi:05d}.jpg"

            shutil.copy(frame_path, os.path.join(out_dir, 'images', img_name))

            if fi not in all_results:

                open(os.path.join(out_dir, 'labels', f'{fi:05d}.txt'), 'w').close()
                shutil.copy(frame_path, os.path.join(out_dir, 'images_vis', img_name))
                shutil.copy(frame_path,
                            os.path.join(out_dir, 'images_vis_template_only', img_name))
                with open(os.path.join(out_dir, 'keypoints', f'{fi:05d}.txt'), 'w') as f:
                    f.write("0 0 0 0 0 0\n")
                skipped += 1
                continue

            r = all_results[fi]

            align = align_from_anchor(
                tpl_endpoints, tpl_center, bottom_key,
                r["bottom_pos"], scale, r["angle_deg"])
            centroid = (align["pos_x"], align["pos_y"])

            warped = warp_template(
                tpl_binary, tpl_center,
                align["pos_x"], align["pos_y"],
                scale, r["angle_deg"], fw, fh,
                dilate_px=dilate_px)

            lbl_path = os.path.join(out_dir, 'labels', f'{fi:05d}.txt')
            if np.any(warped > 0):
                poly = mask_to_yolo_seg(warped, fw, fh)
                if poly and len(poly) >= 6:
                    pa = np.array(poly)
                    if not (np.any(pa < 0) or np.any(pa > 1)):
                        with open(lbl_path, 'w') as f:
                            f.write(f"{class_id} "
                                    f"{' '.join(f'{c:.6f}' for c in poly)}\n")
                    else:
                        open(lbl_path, 'w').close()
                        skipped += 1
                        continue
                else:
                    open(lbl_path, 'w').close()
                    skipped += 1
                    continue
            else:
                open(lbl_path, 'w').close()
                skipped += 1
                continue

            vis_rgb = build_vis_with_markers(
                frame_bgr, warped,
                r["bottom_pos"], r["top_pos"],
                centroid, r["method"])
            cv2.imwrite(
                os.path.join(out_dir, 'images_vis', img_name),
                cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))

            tpl_vis = build_template_only_vis(frame_bgr, warped)
            cv2.imwrite(
                os.path.join(out_dir, 'images_vis_template_only', img_name),
                cv2.cvtColor(tpl_vis, cv2.COLOR_RGB2BGR))

            with open(os.path.join(out_dir, 'keypoints', f'{fi:05d}.txt'), 'w') as f:
                f.write(f"{r['bottom_pos'][0]:.2f} {r['bottom_pos'][1]:.2f} "
                        f"{r['top_pos'][0]:.2f} {r['top_pos'][1]:.2f} "
                        f"{centroid[0]:.2f} {centroid[1]:.2f}\n")

            saved += 1

        import csv as csv_mod
        csv_path = os.path.join(out_dir, 'tracking_results.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv_mod.writer(csvf)
            writer.writerow([
                'frame', 'bottom_x', 'bottom_y', 'top_x', 'top_y',
                'centroid_x', 'centroid_y', 'angle_deg',
                'endpoint_dist', 'method',
            ])
            for fi in range(len(frames)):
                if fi not in all_results:
                    writer.writerow([fi, 0, 0, 0, 0, 0, 0, 0, 0, 'missing'])
                    continue
                r = all_results[fi]
                bx, by = r["bottom_pos"]
                tx, ty = r["top_pos"]
                a_align = align_from_anchor(
                    tpl_endpoints, tpl_center, bottom_key,
                    r["bottom_pos"], scale, r["angle_deg"])
                cx, cy = a_align["pos_x"], a_align["pos_y"]
                ep_dist = np.sqrt((tx-bx)**2 + (ty-by)**2)
                writer.writerow([
                    fi,
                    f'{bx:.2f}', f'{by:.2f}',
                    f'{tx:.2f}', f'{ty:.2f}',
                    f'{cx:.2f}', f'{cy:.2f}',
                    f'{r["angle_deg"]:.2f}',
                    f'{ep_dist:.2f}',
                    r["method"],
                ])
        log(f"\n📊 汇总CSV已保存: {csv_path}")

        progress(1.0, desc="完成!")
        log(f"\n{'=' * 50}")
        log(f"   导出完成!")
        log(f"   总帧数: {len(frames)}")
        log(f"   有效帧: {saved}")
        log(f"   空标注: {skipped}")
        log(f"   输出目录: {out_dir}")
        log(f"     ├── images/                  (原图)")
        log(f"     ├── labels/                  (YOLO 标注)")
        log(f"     ├── images_vis/              (模板+端点标记)")
        log(f"     ├── images_vis_template_only/ (纯模板叠加)")
        log(f"     ├── keypoints/               (每帧端点+质心 txt)")
        log(f"     └── tracking_results.csv     (汇总CSV)")
        log(f"\n关键点 txt 格式 (每帧一个文件):")
        log(f"  bottom_x bottom_y top_x top_y centroid_x centroid_y")
        log(f"\nCSV 列说明:")
        log(f"  frame        - 帧序号")
        log(f"  bottom_x/y   - 底部端点圆质心 (px)")
        log(f"  top_x/y      - 顶部端点圆质心 (px)")
        log(f"  centroid_x/y - 整个目标质心 (px)")
        log(f"  angle_deg    - 模板旋转角 (度)")
        log(f"  endpoint_dist- 两端点间距 (px)")
        log(f"  method       - 追踪方法")

        _tracking_results["vname"] = video_dir_name
        _tracking_results["results"] = all_results
        _tracking_results["frames"] = frames
        _tracking_results["out_dir"] = out_dir

        first_vis = os.path.join(out_dir, 'images_vis',
                                 f'{start_fi:05d}.jpg')
        first_img = None
        if os.path.exists(first_vis):
            first_img = cv2.cvtColor(cv2.imread(first_vis), cv2.COLOR_BGR2RGB)

        return "\n".join(log_lines), first_img

    except Exception as e:
        traceback.print_exc()
        return f"❌ 失败: {e}\n{traceback.format_exc()}", None

def on_browse(video_dir_name, frame_idx):

    if not _tracking_results or _tracking_results.get("vname") != video_dir_name:
        return None, "请先运行追踪"
    try:
        out_dir = _tracking_results["out_dir"]
        fi = int(frame_idx)
        vis_path = os.path.join(out_dir, 'images_vis', f'{fi:05d}.jpg')
        if os.path.exists(vis_path):
            vis = cv2.cvtColor(cv2.imread(vis_path), cv2.COLOR_BGR2RGB)
            r = _tracking_results["results"].get(fi, {})
            info = (f"帧 {fi} | 方法: {r.get('method', 'N/A')} | "
                    f"角度: {r.get('angle_deg', 0):.1f}°")
            return vis, info
        return None, f"帧 {fi} 可视化不存在"
    except Exception as e:
        return None, f"错误: {e}"

def build_app():
    dirs = list_video_dirs()

    with gr.Blocks(title="光流模板追踪") as app:
        gr.Markdown("# ⚡ 光流追踪 + 模板对齐工具")
        gr.Markdown(
            "**Step 1**: 手动对齐模板到指定帧 (点击定位 + 缩放/旋转/膨胀)\n\n"
            "**Step 2**: LK 稀疏光流追踪端点 → 逐帧 warp 模板 → "
            "导出 YOLO 标注 + 关键点坐标"
        )

        with gr.Accordion("Step 1: 模板对齐 (手动)", open=True):
            with gr.Row():
                video_dir_dd = gr.Dropdown(
                    choices=dirs, label="视频帧目录",
                    info="video_to_img/ 下的子文件夹")
                template_path_input = gr.Textbox(
                    value=DEFAULT_TEMPLATE, label="模板图片路径")
                start_frame_input = gr.Number(
                    value=0, label="起始帧", precision=0,
                    info="Step1 对齐所用的帧 (也是光流追踪起点)")
                load_btn = gr.Button("加载数据", variant="primary")

            pos_x_state = gr.State(None)
            pos_y_state = gr.State(None)

            with gr.Row():
                with gr.Column(scale=1):
                    tpl_img = gr.Image(label="模板轮廓", interactive=False,
                                       height=220)
                    gr.Markdown("📍 **点击右图定位** | 🎚 滑条调缩放/旋转")
                    scale_slider = gr.Slider(
                        0, 0.5, 0.07, step=0.001, label="缩放",
                        info="刚体不变形, 全程固定")
                    angle_slider = gr.Slider(
                        -180, 180, -92.5, step=0.5, label="旋转 (度)")
                    dilate_slider = gr.Slider(
                        0, 10, 3, step=1, label="模板膨胀 (px)",
                        info="加粗中间细杆, 0=不膨胀")
                    alpha_slider = gr.Slider(
                        10, 80, 40, step=5, label="透明度 %")
                    with gr.Row():
                        update_btn = gr.Button("刷新", variant="secondary")
                    info_box = gr.Textbox(
                        label="状态", lines=3, interactive=False)

                with gr.Column(scale=2):
                    preview_img = gr.Image(
                        label="点击设置位置 | 绿线=轮廓 | 青色=区域 | 红十字=中心",
                        interactive=False)
                    preview_info = gr.Textbox(
                        label="预览", lines=1, interactive=False)

        with gr.Accordion("Step 2: 光流追踪 + 导出", open=True):
            gr.Markdown(
                "从 Step 1 的端点位置出发，用 **LK 稀疏光流**逐帧追踪两个端点 → "
                "计算角度 → warp 模板 → 导出\n\n"
                "🟢 B = 底部端点 | 🔵 T = 顶部端点 | 🔴 C = 质心"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    s2_class_id = gr.Number(
                        value=0, label="CLASS_ID", precision=0)
                    s2_track_angle = gr.Checkbox(
                        value=True, label="角度追踪",
                        info="从双端点动态计算旋转角度")
                    s2_run_btn = gr.Button(
                        "🚀 光流追踪 + 导出", variant="primary", size="lg")

                with gr.Column(scale=2):
                    s2_log = gr.Textbox(
                        label="运行日志", lines=18, interactive=False)

            gr.Markdown("#### 浏览结果")
            with gr.Row():
                s2_browse_slider = gr.Slider(
                    0, 1000, 0, step=1, label="帧索引")
                s2_browse_btn = gr.Button("查看", variant="secondary")
            s2_browse_img = gr.Image(
                label="追踪可视化", interactive=False)
            s2_browse_info = gr.Textbox(
                label="帧信息", lines=1, interactive=False)

        load_btn.click(
            fn=on_load_data,
            inputs=[video_dir_dd, template_path_input, start_frame_input],
            outputs=[tpl_img, preview_img, pos_x_state, pos_y_state,
                     scale_slider, info_box])

        preview_img.select(
            fn=on_click_frame,
            inputs=[pos_x_state, pos_y_state, video_dir_dd,
                    template_path_input, start_frame_input,
                    scale_slider, angle_slider, dilate_slider, alpha_slider],
            outputs=[preview_img, pos_x_state, pos_y_state, preview_info])

        _s_in = [pos_x_state, pos_y_state, video_dir_dd, template_path_input,
                 start_frame_input,
                 scale_slider, angle_slider, dilate_slider, alpha_slider]
        _s_out = [preview_img, preview_info]
        for ctrl in [scale_slider, angle_slider, dilate_slider, alpha_slider]:
            ctrl.release(fn=on_slider_change, inputs=_s_in, outputs=_s_out)
        update_btn.click(fn=on_slider_change, inputs=_s_in, outputs=_s_out)

        def on_vdir_change(vname):
            if not vname:
                return gr.update()
            n = len(get_sorted_frames(vname))
            return gr.update(maximum=max(n - 1, 0))

        video_dir_dd.change(
            fn=on_vdir_change,
            inputs=[video_dir_dd],
            outputs=[s2_browse_slider])

        s2_run_btn.click(
            fn=on_run_tracking,
            inputs=[video_dir_dd, template_path_input, start_frame_input,
                    pos_x_state, pos_y_state, scale_slider, angle_slider,
                    dilate_slider, s2_class_id, s2_track_angle],
            outputs=[s2_log, s2_browse_img])

        s2_browse_btn.click(
            fn=on_browse,
            inputs=[video_dir_dd, s2_browse_slider],
            outputs=[s2_browse_img, s2_browse_info])

    return app

if __name__ == '__main__':
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    app = build_app()
    app.launch(
        server_name='0.0.0.0',
        server_port=7863,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
