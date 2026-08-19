#!/usr/bin/env python3

import os
import sys
import shutil
import subprocess
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
import gradio as gr
import traceback
import csv
import glob
from pathlib import Path

from sam2.build_sam import build_sam2_video_predictor
from iros.paths import (
    ASSET_ROOT,
    CHECKPOINT_ROOT,
    FRAME_ROOT,
    OUTPUT_ROOT,
    SHARED_CONTROL_DATA_ROOT,
    VIDEO_ROOT,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = str(FRAME_ROOT)
VIDEOS_DIR = str(VIDEO_ROOT)
YOLO_DATASET_DIR = str(OUTPUT_ROOT / "yolo_datasets")
SAM2_CHECKPOINT = str(CHECKPOINT_ROOT / "sam2.1_hiera_tiny.pt")
MODEL_CFG = 'configs/sam2.1/sam2.1_hiera_t.yaml'

DEFAULT_TEMPLATE_PATH = str(ASSET_ROOT / "robot_template.jpg")

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

def mask_to_yolo_seg(mask, img_w, img_h, simplify_tolerance=2.0):
    mask_2d = np.squeeze(mask)
    if mask_2d.ndim != 2:
        return None
    mask_uint8 = (mask_2d > 0).astype(np.uint8) * 255
    mask_h, mask_w = mask_2d.shape
    if mask_h != img_h or mask_w != img_w:
        mask_uint8 = cv2.resize(mask_uint8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
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

def mask_to_yolo_bbox(mask, img_w, img_h):
    mask_2d = np.squeeze(mask)
    if mask_2d.ndim != 2:
        return None
    mask_h, mask_w = mask_2d.shape
    coords = np.column_stack(np.where(mask_2d > 0))
    if len(coords) == 0:
        return None
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    x_min_mask, x_max_mask = float(x_coords.min()), float(x_coords.max())
    y_min_mask, y_max_mask = float(y_coords.min()), float(y_coords.max())
    if mask_h != img_h or mask_w != img_w:
        scale_x = img_w / mask_w
        scale_y = img_h / mask_h
        x_min = x_min_mask * scale_x
        x_max = x_max_mask * scale_x
        y_min = y_min_mask * scale_y
        y_max = y_max_mask * scale_y
    else:
        x_min, x_max = x_min_mask, x_max_mask
        y_min, y_max = y_min_mask, y_max_mask
    return int(x_min), int(y_min), int(x_max), int(y_max)

def load_template_silhouette(template_path):

    img = cv2.imread(template_path)
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bin_norm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def border_sum(b):
        return (np.sum(b[0, :] > 0) + np.sum(b[-1, :] > 0) +
                np.sum(b[:, 0] > 0) + np.sum(b[:, -1] > 0))
    binary = bin_inv if border_sum(bin_inv) < border_sum(bin_norm) else bin_norm

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary, None
    largest = max(contours, key=cv2.contourArea)

    clean = np.zeros_like(binary)
    cv2.drawContours(clean, [largest], -1, 255, -1)

    return clean, largest

def get_mask_pose(mask_uint8):

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    if M["m00"] < 1:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    area = cv2.contourArea(contour)

    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

    rect = cv2.minAreaRect(contour)

    return {
        "cx": cx, "cy": cy,
        "angle": angle,
        "area": area,
        "scale": np.sqrt(area),
        "rect": rect,
        "contour": contour,
    }

def warp_template_to_pose(template_binary, template_pose, target_cx, target_cy,
                           target_angle, output_size, scale_ratio):

    angle_diff = target_angle - template_pose["angle"]
    cos_a = np.cos(angle_diff)
    sin_a = np.sin(angle_diff)

    tcx, tcy = template_pose["cx"], template_pose["cy"]

    M = np.array([
        [scale_ratio * cos_a, -scale_ratio * sin_a,
         target_cx - scale_ratio * (cos_a * tcx - sin_a * tcy)],
        [scale_ratio * sin_a,  scale_ratio * cos_a,
         target_cy - scale_ratio * (sin_a * tcx + cos_a * tcy)],
    ], dtype=np.float64)

    w, h = output_size
    warped = cv2.warpAffine(template_binary, M, (w, h),
                             flags=cv2.INTER_NEAREST,
                             borderValue=0)
    return warped

def _normalize_mask_to_image(mask, img_w, img_h):

    m = np.squeeze(mask)
    if m.ndim != 2:
        return None
    m_uint8 = (m > 0).astype(np.uint8) * 255
    mh, mw = m_uint8.shape
    if mh != img_h or mw != img_w:
        m_uint8 = cv2.resize(m_uint8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    return m_uint8

def complete_video_masks(video_segments, template_path, ref_frame_idx,
                          img_w, img_h, completeness_thresh=0.7,
                          angle_smooth_alpha=0.5,
                          log_fn=None):

    if log_fn is None:
        log_fn = print

    template_binary, template_contour = load_template_silhouette(template_path)
    if template_binary is None:
        return video_segments, "❌ 模板图片加载失败, 请检查路径"
    template_pose = get_mask_pose(template_binary)
    if template_pose is None:
        return video_segments, "❌ 模板轮廓提取失败, 请检查模板图片"
    log_fn(f"[模板] 尺寸: {template_binary.shape[1]}x{template_binary.shape[0]}, "
           f"面积: {template_pose['area']:.0f}px²")

    ref_mask_uint8 = None
    ref_area = 0
    ref_obj_id = None
    if ref_frame_idx in video_segments:
        for obj_id, mask in video_segments[ref_frame_idx].items():
            m = _normalize_mask_to_image(mask, img_w, img_h)
            if m is not None and np.sum(m > 0) > 0:
                ref_mask_uint8 = m
                ref_area = np.sum(m > 0)
                ref_obj_id = obj_id
                break

    if ref_mask_uint8 is None or ref_area == 0:
        return video_segments, "❌ 参考帧无有效mask, 无法建立映射"

    ref_pose = get_mask_pose(ref_mask_uint8)
    if ref_pose is None:
        return video_segments, "❌ 参考帧mask姿态提取失败"

    scale_ratio = ref_pose["scale"] / max(template_pose["scale"], 1e-6)
    log_fn(f"[参考帧 {ref_frame_idx}] 面积: {ref_area}px, "
           f"缩放比: {scale_ratio:.4f}, 方向: {np.degrees(ref_pose['angle']):.1f}°")

    completed_segments = {}
    n_completed = 0
    n_kept = 0
    n_empty = 0
    prev_angle = ref_pose["angle"]          

    for fi in sorted(video_segments.keys()):
        completed_segments[fi] = {}
        for obj_id, mask in video_segments[fi].items():
            m_uint8 = _normalize_mask_to_image(mask, img_w, img_h)
            if m_uint8 is None:
                completed_segments[fi][obj_id] = mask
                continue

            current_area = np.sum(m_uint8 > 0)

            if current_area == 0:
                completed_segments[fi][obj_id] = mask
                n_empty += 1
                continue

            completeness = current_area / max(ref_area, 1)

            if completeness >= completeness_thresh:
                completed_segments[fi][obj_id] = mask
                n_kept += 1

                cur_pose = get_mask_pose(m_uint8)
                if cur_pose is not None:
                    prev_angle = cur_pose["angle"]
                continue

            cur_pose = get_mask_pose(m_uint8)
            if cur_pose is None:
                completed_segments[fi][obj_id] = mask
                n_empty += 1
                continue

            target_cx = cur_pose["cx"]
            target_cy = cur_pose["cy"]

            raw_angle = cur_pose["angle"]

            angle_diff = (raw_angle - prev_angle + np.pi) % (2 * np.pi) - np.pi
            smoothed_angle = prev_angle + angle_smooth_alpha * angle_diff
            target_angle = smoothed_angle

            warped = warp_template_to_pose(
                template_binary, template_pose,
                target_cx, target_cy, target_angle,
                (img_w, img_h), scale_ratio
            )

            overlap = np.sum((warped > 0) & (m_uint8 > 0))
            overlap_ratio = overlap / max(current_area, 1)

            if overlap_ratio > 0.25:

                merged = np.maximum(warped, m_uint8)
                completed_segments[fi][obj_id] = (merged > 0).astype(np.uint8)
                n_completed += 1
                prev_angle = target_angle
                log_fn(f"  帧 {fi}: ✅ 补全 "
                       f"(完整度={completeness:.1%}, 重叠率={overlap_ratio:.1%})")
            else:

                completed_segments[fi][obj_id] = mask
                n_kept += 1
                log_fn(f"  帧 {fi}: ⚠️ 跳过补全 "
                       f"(重叠率={overlap_ratio:.1%} 太低, 对齐可能失败)")

    info = (f"✅ 模板补全完成!\n"
            f"   总帧: {len(video_segments)}\n"
            f"   补全: {n_completed} 帧\n"
            f"   保留原始: {n_kept} 帧\n"
            f"   空mask: {n_empty} 帧\n"
            f"   完整度阈值: {completeness_thresh:.0%}")
    log_fn(info)
    return completed_segments, info

def preview_template_extraction(template_path):

    if not template_path or not os.path.exists(template_path):
        return None, "模板文件不存在"
    binary, contour = load_template_silhouette(template_path)
    if binary is None:
        return None, "模板加载失败"

    img = cv2.imread(template_path)
    if img is None:
        return None, "图片读取失败"

    left = img.copy()
    if contour is not None:
        cv2.drawContours(left, [contour], -1, (0, 255, 0), 3)
        pose = get_mask_pose(binary)
        if pose:

            cx, cy = int(pose["cx"]), int(pose["cy"])
            cv2.drawMarker(left, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            length = 60
            dx = int(length * np.cos(pose["angle"]))
            dy = int(length * np.sin(pose["angle"]))
            cv2.arrowedLine(left, (cx, cy), (cx + dx, cy + dy),
                            (255, 0, 0), 2, tipLength=0.3)

    right = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    if h1 != h2:
        scale = h1 / h2
        right = cv2.resize(right, (int(w2 * scale), h1))

    vis = np.hstack([left, right])
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    area = int(np.sum(binary > 0))
    info = (f"模板提取成功!\n"
            f"  轮廓点数: {len(contour) if contour is not None else 0}\n"
            f"  物体面积: {area} px²\n"
            f"  左=原图+轮廓(绿), 右=二值mask\n"
            f"  红十字=中心, 蓝箭头=主轴方向")
    return Image.fromarray(vis_rgb), info

def list_video_files():

    if not os.path.isdir(VIDEOS_DIR):
        return []
    exts = ('.mp4', '.avi', '.mov', '.mkv')
    return sorted([f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(exts)])

def on_extract_frames(video_file, frame_interval):

    if not video_file:
        return "请先选择视频文件", gr.update()
    video_path = os.path.join(VIDEOS_DIR, video_file)
    output_name = Path(video_file).stem
    output_path = os.path.join(BASE_DIR, output_name)
    os.makedirs(output_path, exist_ok=True)
    frame_interval = max(1, int(frame_interval))
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select='not(mod(n,{frame_interval}))'",
        "-vsync", "0", "-q:v", "2", "-start_number", "0",
        os.path.join(output_path, "%05d.jpg"),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                       encoding='utf-8', errors='ignore')
        n = len([f for f in os.listdir(output_path) if f.lower().endswith('.jpg')])
        new_dirs = list_video_dirs()
        return (f"切帧完成: {output_name}, 共 {n} 帧 (每 {frame_interval} 帧取 1 帧)\n"
                f"保存至: {output_path}"),\
               gr.update(choices=new_dirs, value=output_name)
    except Exception as e:
        return f"切帧失败: {e}", gr.update()

def list_video_dirs():

    dirs = []
    if not os.path.isdir(BASE_DIR):
        return dirs
    for d in sorted(os.listdir(BASE_DIR)):
        full = os.path.join(BASE_DIR, d)
        if os.path.isdir(full):
            jpgs = [f for f in os.listdir(full) if f.lower().endswith(('.jpg', '.jpeg'))]
            if jpgs:
                dirs.append(d)
    return dirs

def get_sorted_frame_names(video_dir_name):

    d = os.path.join(BASE_DIR, video_dir_name)
    names = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg'))]
    names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return names

def load_frame_image(video_dir_name, frame_idx):

    names = get_sorted_frame_names(video_dir_name)
    if not names:
        return None
    frame_idx = max(0, min(frame_idx, len(names) - 1))
    path = os.path.join(BASE_DIR, video_dir_name, names[frame_idx])
    return Image.open(path).convert('RGB')

def draw_points_on_image(pil_img, points, labels):

    img_draw = pil_img.copy()
    draw = ImageDraw.Draw(img_draw)
    r = 6
    for i, (pt, lbl) in enumerate(zip(points, labels)):
        x, y = int(pt[0]), int(pt[1])
        color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
        outline = (255, 255, 255)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=outline, width=2)

        draw.line([(x - r - 2, y), (x + r + 2, y)], fill=outline, width=1)
        draw.line([(x, y - r - 2), (x, y + r + 2)], fill=outline, width=1)

        draw.text((x + r + 4, y - r), str(i + 1), fill='white')
    return img_draw

def draw_mask_overlay(pil_img, mask, alpha=0.45, color=(255, 50, 50)):

    img_np = np.array(pil_img).copy()
    mask_2d = np.squeeze(mask)
    if mask_2d.ndim != 2:
        return pil_img

    h, w = img_np.shape[:2]
    mh, mw = mask_2d.shape
    if mh != h or mw != w:
        mask_2d = cv2.resize(mask_2d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = img_np.copy()
    overlay[mask_2d > 0] = list(color)
    img_np = (img_np * (1 - alpha) + overlay * alpha).astype(np.uint8)

    contours, _ = cv2.findContours(
        (mask_2d > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img_np, contours, -1, (0, 255, 0), 2)
    return Image.fromarray(img_np)

def on_video_dir_change(video_dir_name):

    if not video_dir_name:
        return None, 0, gr.update(maximum=0), "请选择视频目录", [], [], None
    names = get_sorted_frame_names(video_dir_name)
    total = len(names)
    img = load_frame_image(video_dir_name, 0)
    info = f"📂 {video_dir_name} — 共 {total} 帧，图片尺寸: {img.size[0]}x{img.size[1]}"
    return img, 0, gr.update(maximum=max(total - 1, 0)), info, [], [], None

def on_frame_change(video_dir_name, frame_idx, points_state, labels_state):

    if not video_dir_name:
        return None
    img = load_frame_image(video_dir_name, frame_idx)
    if img is None:
        return None
    if points_state:
        img = draw_points_on_image(img, points_state, labels_state)
    return img

def on_image_click(video_dir_name, frame_idx, point_type, points_state, labels_state, evt: gr.SelectData):

    if not video_dir_name:
        return None, points_state, labels_state, "请先选择视频目录"

    x, y = evt.index[0], evt.index[1]
    label = 1 if point_type == "正样本 (前景)" else 0

    points_state.append([x, y])
    labels_state.append(label)

    img = load_frame_image(video_dir_name, frame_idx)
    img = draw_points_on_image(img, points_state, labels_state)

    lines = []
    for i, (pt, lbl) in enumerate(zip(points_state, labels_state)):
        tag = "正样本" if lbl == 1 else "负样本"
        lines.append(f"  {i + 1}. ({int(pt[0])}, {int(pt[1])})  {tag}")
    info = f"已选 {len(points_state)} 个点：\n" + "\n".join(lines)

    return img, points_state, labels_state, info

def on_clear_points(video_dir_name, frame_idx):

    img = load_frame_image(video_dir_name, frame_idx) if video_dir_name else None
    return img, [], [], "已清除所有点"

def on_undo_point(video_dir_name, frame_idx, points_state, labels_state):

    if points_state:
        points_state.pop()
        labels_state.pop()
    img = load_frame_image(video_dir_name, frame_idx) if video_dir_name else None
    if img is not None and points_state:
        img = draw_points_on_image(img, points_state, labels_state)

    if points_state:
        lines = []
        for i, (pt, lbl) in enumerate(zip(points_state, labels_state)):
            tag = "正样本" if lbl == 1 else "负样本"
            lines.append(f"  {i + 1}. ({int(pt[0])}, {int(pt[1])})  {tag}")
        info = f"已选 {len(points_state)} 个点：\n" + "\n".join(lines)
    else:
        info = "已清除所有点"
    return img, points_state, labels_state, info

def on_preview_mask(video_dir_name, frame_idx, points_state, labels_state):

    if not video_dir_name:
        return None, "请先选择视频目录"
    if not points_state:
        return None, "请先在图片上点击选择至少一个标记点"

    try:
        predictor = get_predictor()
        video_path = os.path.join(BASE_DIR, video_dir_name)
        inference_state = predictor.init_state(video_path=video_path)

        points_np = np.array(points_state, dtype=np.float32)
        labels_np = np.array(labels_state, dtype=np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=int(frame_idx),
            obj_id=1,
            points=points_np,
            labels=labels_np,
        )

        mask = (out_mask_logits[0] > 0.0).cpu().numpy()

        img = load_frame_image(video_dir_name, frame_idx)
        result_img = draw_mask_overlay(img, mask)
        result_img = draw_points_on_image(result_img, points_state, labels_state)

        predictor.reset_state(inference_state)

        mask_pixels = int(np.sum(np.squeeze(mask) > 0))
        return result_img, f"Mask 预览成功！mask 像素数: {mask_pixels}"
    except Exception as e:
        traceback.print_exc()
        return None, f"预览失败: {str(e)}"

def on_export_yolo(video_dir_name, frame_idx, points_state, labels_state, class_id,
                   enable_completion, template_path, completion_thresh,
                   angle_smooth, progress=gr.Progress()):

    if not video_dir_name:
        return "请先选择视频目录"
    if not points_state:
        return "请先在图片上点击选择至少一个标记点"

    try:
        log_lines = []

        def log(msg):
            log_lines.append(msg)
            print(msg)

        predictor = get_predictor()
        video_path = os.path.join(BASE_DIR, video_dir_name)

        log(f"视频目录: {video_path}")
        log(f"设备: {get_device()}")
        log(f"标记帧: {frame_idx}, 选点数: {len(points_state)}, CLASS_ID: {class_id}")
        if enable_completion:
            log(f"🔧 模板补全已启用: {template_path}")
            log(f"   完整度阈值: {completion_thresh}, 角度平滑: {angle_smooth}")

        progress(0.0, desc="初始化 SAM2 ...")
        inference_state = predictor.init_state(video_path=video_path)

        points_np = np.array(points_state, dtype=np.float32)
        labels_np = np.array(labels_state, dtype=np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=int(frame_idx),
            obj_id=1,
            points=points_np,
            labels=labels_np,
        )
        log(f"已添加提示点，目标数: {len(out_obj_ids)}")

        progress(0.1, desc="正向传播中 ...")
        video_segments = {}
        for out_frame_idx, out_obj_ids_prop, out_mask_logits_prop in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits_prop[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids_prop)
            }
        log(f"传播完成: {len(video_segments)} 帧有 mask")

        predictor.reset_state(inference_state)

        if enable_completion and template_path and os.path.exists(template_path):
            progress(0.3, desc="模板对齐补全中 ...")

            frame_names_tmp = get_sorted_frame_names(video_dir_name)
            sample_path = os.path.join(video_path, frame_names_tmp[0])
            sample_img = Image.open(sample_path)
            _iw, _ih = sample_img.size

            log(f"\n{'─' * 40}")
            log(f"开始模板补全 (图片尺寸: {_iw}x{_ih})")

            video_segments, comp_info = complete_video_masks(
                video_segments,
                template_path=template_path,
                ref_frame_idx=int(frame_idx),
                img_w=_iw, img_h=_ih,
                completeness_thresh=float(completion_thresh),
                angle_smooth_alpha=float(angle_smooth),
                log_fn=log,
            )
            log(f"{'─' * 40}\n")
        elif enable_completion:
            log(f"⚠️ 模板补全已启用但模板文件不存在: {template_path}")

        progress(0.4, desc="导出 YOLO 数据集 ...")
        output_dir = os.path.join(YOLO_DATASET_DIR, video_dir_name)
        images_dir = os.path.join(output_dir, 'images')
        labels_dir = os.path.join(output_dir, 'labels')
        vis_dir = os.path.join(output_dir, 'images_vis')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        frame_names = get_sorted_frame_names(video_dir_name)
        total_frames = len(frame_names)
        saved_count = 0
        skipped_count = 0
        class_id_int = int(class_id)

        for fi in range(total_frames):
            progress(0.4 + 0.55 * (fi / total_frames), desc=f"处理帧 {fi + 1}/{total_frames} ...")

            img_path = os.path.join(video_path, frame_names[fi])
            img = Image.open(img_path).convert('RGB')
            img_w, img_h = img.size

            label_lines = []
            bboxes_px = []

            if fi in video_segments:
                for obj_id, mask in video_segments[fi].items():
                    mask_2d = np.squeeze(mask)
                    if mask_2d.ndim != 2 or not np.any(mask_2d > 0):
                        continue
                    polygon = mask_to_yolo_seg(mask_2d, img_w, img_h, simplify_tolerance=2.0)
                    if polygon is None or len(polygon) < 6:
                        continue
                    polygon_array = np.array(polygon)
                    if np.any(polygon_array < 0) or np.any(polygon_array > 1):
                        continue
                    polygon_str = ' '.join([f'{coord:.6f}' for coord in polygon])
                    label_lines.append(f"{class_id_int} {polygon_str}\n")
                    bbox = mask_to_yolo_bbox(mask_2d, img_w, img_h)
                    if bbox is not None:
                        bboxes_px.append(bbox)

            img_name = f'{fi:05d}.jpg'
            shutil.copy(img_path, os.path.join(images_dir, img_name))

            label_name = f'{fi:05d}.txt'
            with open(os.path.join(labels_dir, label_name), 'w', encoding='utf-8') as f:
                if label_lines:
                    f.writelines(label_lines)

            if not label_lines:
                skipped_count += 1

            fig = plt.figure(figsize=(6, 4), dpi=100)
            ax = plt.gca()
            ax.axis('off')
            ax.set_title(f'frame {fi}')
            ax.imshow(img)
            if fi in video_segments:
                for obj_id, mask in video_segments[fi].items():
                    mask_2d = np.squeeze(mask)
                    if mask_2d.ndim != 2 or not np.any(mask_2d > 0):
                        continue

                    m = mask_2d.astype(np.float32)
                    h, w = m.shape
                    rgba = np.zeros((h, w, 4), dtype=np.float32)
                    rgba[..., 0] = 1.0
                    rgba[..., 1] = 0.2
                    rgba[..., 2] = 0.2
                    rgba[..., 3] = m * 0.45
                    ax.imshow(rgba)
                    for bbox in bboxes_px:
                        x_min, y_min, x_max, y_max = bbox
                        rect = Rectangle(
                            (x_min, y_min), x_max - x_min + 1, y_max - y_min + 1,
                            fill=False, linewidth=2, edgecolor='green',
                        )
                        ax.add_patch(rect)
            plt.savefig(os.path.join(vis_dir, img_name), bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            saved_count += 1

        progress(1.0, desc="完成！")

        log(f"\n{'=' * 50}")
        log(f"   导出完成！")
        log(f"   总帧数: {total_frames}")
        log(f"   有 mask 的帧: {saved_count - skipped_count}")
        log(f"   无 mask 的帧: {skipped_count}")
        if enable_completion:
            log(f"   (已启用模板补全)")
        log(f"   输出目录: {output_dir}")
        log(f"     ├── images/     (原图)")
        log(f"     ├── labels/     (YOLO 实例分割label)")
        log(f"     └── images_vis/ (可视化)")

        return "\n".join(log_lines)

    except Exception as e:
        traceback.print_exc()
        return f"导出失败: {str(e)}\n{traceback.format_exc()}"

def list_yolo_datasets():

    if not os.path.isdir(YOLO_DATASET_DIR):
        return []
    return sorted([
        d for d in os.listdir(YOLO_DATASET_DIR)
        if os.path.isdir(os.path.join(YOLO_DATASET_DIR, d, 'images'))
    ])

SC_DIR = str(SHARED_CONTROL_DATA_ROOT)
SC_OUT_DIR = os.path.join(SC_DIR, 'registration_output')
RAY_MAX = 200

def _load_maze_rgb(maze_path):

    m = cv2.imread(maze_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.shape[2] == 4:
        a = m[:, :, 3:4].astype(np.float32) / 255
        bgr = m[:, :, :3].astype(np.float32)
        rgb = (bgr * a + 255 * (1 - a)).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(m, cv2.COLOR_BGR2RGB)

def _draw_pts(img, pts, colors=((255,0,0),(0,200,0),(0,0,255),(255,165,0))):
    vis = img.copy()
    h, w = vis.shape[:2]
    r = max(5, min(h, w) // 60)
    for i, (x, y) in enumerate(pts):
        c = colors[i % len(colors)]
        cv2.circle(vis, (int(x), int(y)), r, c, -1)
        cv2.circle(vis, (int(x), int(y)), r, (255,255,255), 2)
        cv2.putText(vis, str(i+1), (int(x)+r+2, int(y)+r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    return vis

def sc_on_dataset_change(dataset_name):

    if not dataset_name:
        return None, None, None, "请选择数据集"
    src = os.path.join(YOLO_DATASET_DIR, dataset_name)

    for sub in ("images_vis", "labels"):
        dst = os.path.join(SC_DIR, sub)
        src_sub = os.path.join(src, sub)
        if os.path.isdir(src_sub):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src_sub, dst)

    imgs = sorted(glob.glob(os.path.join(SC_DIR, "images_vis", "*.jpg")))
    if not imgs:
        return None, None, None, "未找到图像"
    cam = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2RGB)
    img_h, img_w = cam.shape[:2]
    n_labels = len(glob.glob(os.path.join(SC_DIR, "labels", "*.txt")))

    maze_path = os.path.join(SC_DIR, "maze1.png")
    maze_img = _load_maze_rgb(maze_path) if os.path.exists(maze_path) else None
    info = f"已加载 {len(imgs)} 帧, {n_labels} 标签, 尺寸 {img_w}x{img_h}"
    if maze_img is not None:
        info += f"\n迷宫图已加载: {maze_path}"
    else:
        info += f"\n迷宫图不存在: {maze_path}"
    return maze_img, cam, None, info

def sc_load_maze():

    maze_path = os.path.join(SC_DIR, "maze1.png")
    if not os.path.exists(maze_path):
        return None, f"迷宫图不存在: {maze_path}"
    rgb = _load_maze_rgb(maze_path)
    return rgb, f"迷宫图已加载: {maze_path}"

def sc_click_maze(pts, evt: gr.SelectData):
    x, y = evt.index
    if len(pts) >= 4:
        pts = []
    pts.append([x, y])
    maze_path = os.path.join(SC_DIR, "maze1.png")
    rgb = _load_maze_rgb(maze_path)
    info = f"迷宫点 ({len(pts)}/4): " + ", ".join(f"({p[0]},{p[1]})" for p in pts)
    return _draw_pts(rgb, pts) if rgb is not None else None, pts, info

def sc_click_cam(pts, evt: gr.SelectData):
    x, y = evt.index
    if len(pts) >= 4:
        pts = []
    pts.append([x, y])
    imgs = sorted(glob.glob(os.path.join(SC_DIR, "images_vis", "*.jpg")))
    cam = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2RGB) if imgs else None
    info = f"相机点 ({len(pts)}/4): " + ", ".join(f"({p[0]},{p[1]})" for p in pts)
    return _draw_pts(cam, pts) if cam is not None else None, pts, info

def sc_register(maze_pts, cam_pts, alpha_val):

    if len(maze_pts) != 4 or len(cam_pts) != 4:
        return None, "请在两张图上各点击 4 个对应点"
    os.makedirs(SC_OUT_DIR, exist_ok=True)
    maze_path = os.path.join(SC_DIR, "maze1.png")
    imgs = sorted(glob.glob(os.path.join(SC_DIR, "images_vis", "*.jpg")))
    if not imgs:
        return None, "未找到图像"

    src, dst = np.float32(maze_pts), np.float32(cam_pts)
    H, _ = cv2.findHomography(src, dst)

    maze_raw = cv2.imread(maze_path, cv2.IMREAD_UNCHANGED)
    cam_bgr = cv2.imread(imgs[0])
    h, w = cam_bgr.shape[:2]
    bgr = maze_raw[:, :, :3]
    a = maze_raw[:, :, 3] if maze_raw.shape[2] == 4 else np.full(maze_raw.shape[:2], 255, np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    w_alpha = cv2.warpPerspective(a, H, (w, h))
    mask = (w_alpha.astype(np.float32) / 255.0)[..., None] * (alpha_val / 100.0)
    overlay = np.zeros_like(cam_bgr); overlay[:] = (0, 128, 128)
    result = np.clip(cam_bgr * (1 - mask) + overlay * mask, 0, 255).astype(np.uint8)

    wall_src = ((a > 50) & (gray < 100)).astype(np.uint8) * 255
    wall_mask = (cv2.warpPerspective(wall_src, H, (w, h)) > 128).astype(np.uint8) * 255
    corr_src = ((a > 50) & (gray >= 100)).astype(np.uint8) * 255
    corr_mask = (cv2.warpPerspective(corr_src, H, (w, h)) > 128).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(SC_OUT_DIR, "maze_wall_mask.png"), wall_mask)
    cv2.imwrite(os.path.join(SC_OUT_DIR, "maze_corridor_mask.png"), corr_mask)
    cv2.imwrite(os.path.join(SC_OUT_DIR, "registered.png"), result)
    np.save(os.path.join(SC_OUT_DIR, "homography.npy"), H)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result_rgb, f"配准完成!\nH =\n{H}"

def _raycast(cx, cy, dx, dy, mask, hit_val, max_d=RAY_MAX):
    h, w = mask.shape
    for s in range(1, max_d):
        x, y = int(round(cx + dx*s)), int(round(cy + dy*s))
        if x < 0 or x >= w or y < 0 or y >= h:
            return float(s)
        if mask[y, x] == hit_val:
            return float(s)
    return float(max_d)

def _heading_dirs(vx, vy):
    sp = np.sqrt(vx**2 + vy**2)
    if sp < 0.01: vx, vy, sp = 0.0, 1.0, 1.0
    hx, hy = vx/sp, vy/sp
    return (hx, hy), (-hy, hx), (hy, -hx)

def sc_analyze(step_val, progress=gr.Progress()):

    h_path = os.path.join(SC_OUT_DIR, "homography.npy")
    if not os.path.exists(h_path):
        return None, None, "请先完成配准"

    step = max(1, int(step_val))
    maze_path = os.path.join(SC_DIR, "maze1.png")
    maze = cv2.imread(maze_path, cv2.IMREAD_UNCHANGED)
    a, gray = maze[:,:,3], cv2.cvtColor(maze[:,:,:3], cv2.COLOR_BGR2GRAY)
    H = np.load(h_path)

    imgs = sorted(glob.glob(os.path.join(SC_DIR, "images_vis", "*.jpg")))
    sample = cv2.imread(imgs[0])
    img_h, img_w = sample.shape[:2]

    wall_src = ((a > 50) & (gray < 100)).astype(np.uint8) * 255
    wall_mask = (cv2.warpPerspective(wall_src, H, (img_w, img_h)) > 128).astype(np.uint8) * 255
    corr_src = ((a > 50) & (gray >= 100)).astype(np.uint8) * 255
    corr_mask = (cv2.warpPerspective(corr_src, H, (img_w, img_h)) > 128).astype(np.uint8) * 255

    label_dir = os.path.join(SC_DIR, "labels")
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    all_pos = {}
    for lf in label_files:
        fid = int(os.path.splitext(os.path.basename(lf))[0])
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                coords = list(map(float, parts[1:]))
                pts = np.array([[coords[i]*img_w, coords[i+1]*img_h]
                                for i in range(0, len(coords), 2)])
                all_pos[fid] = tuple(pts.mean(axis=0))
                break

    sampled = sorted(fid for fid in all_pos if fid % step == 0)
    if not sampled:
        return None, None, "无有效标签"

    results = []
    prev_hd = None
    for i, fid in enumerate(progress.tqdm(sampled, desc="分析中")):
        cx, cy = all_pos[fid]
        vx = vy = speed = 0.0
        if i > 0:
            px, py = all_pos[sampled[i-1]]
            vx, vy = (cx-px)/step, (cy-py)/step
            speed = np.sqrt(vx**2 + vy**2)

        heading = np.arctan2(vy, vx) if speed > 0.01 else (prev_hd or np.pi/2)
        curv = 0.0
        if prev_hd is not None and speed > 0.01:
            dth = (heading - prev_hd + np.pi) % (2*np.pi) - np.pi
            ds = speed * step
            curv = dth / ds if ds > 0.1 else 0.0
        prev_hd = heading

        _, (lx,ly), (rx,ry) = _heading_dirs(vx, vy)
        dl_w = _raycast(cx, cy, lx, ly, wall_mask, 255)
        dr_w = _raycast(cx, cy, rx, ry, wall_mask, 255)
        dl_c = _raycast(cx, cy, lx, ly, corr_mask, 0)
        dr_c = _raycast(cx, cy, rx, ry, corr_mask, 0)

        results.append({
            "frame": fid,
            "cx": round(float(cx),2), "cy": round(float(cy),2),
            "vx": round(float(vx),3), "vy": round(float(vy),3),
            "speed": round(float(speed),3),
            "heading_deg": round(float(np.degrees(heading)),1),
            "curvature": round(float(curv),4),
            "dist_l_wall": round(float(dl_w),1), "dist_r_wall": round(float(dr_w),1),
            "dist_l_corr": round(float(dl_c),1), "dist_r_corr": round(float(dr_c),1),
            "corridor_w": round(float(dl_c + dr_c),1),
        })

    os.makedirs(SC_OUT_DIR, exist_ok=True)
    csv_path = os.path.join(SC_OUT_DIR, "robot_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    vis = sample.copy()
    c1, _ = cv2.findContours(corr_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, c1, -1, (200,180,0), 1)
    c2, _ = cv2.findContours(wall_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, c2, -1, (0,0,200), 1)

    max_k = max((abs(r["curvature"]) for r in results[1:]), default=0.001) or 0.001
    for j in range(1, len(results)):
        p1 = (int(results[j-1]["cx"]), int(results[j-1]["cy"]))
        p2 = (int(results[j]["cx"]), int(results[j]["cy"]))
        kn = min(abs(results[j]["curvature"]) / max_k, 1.0)
        cv2.line(vis, p1, p2, (0, int(255*(1-kn)), int(255*kn)), 2)
    for j in range(0, len(results), max(1, len(results)//15)):
        r = results[j]
        pt = (int(r["cx"]), int(r["cy"]))
        (hx,hy),_,_ = _heading_dirs(r["vx"], r["vy"])
        cv2.arrowedLine(vis, pt, (int(r["cx"]+hx*15), int(r["cy"]+hy*15)), (255,255,255), 1, tipLength=0.3)
        cv2.circle(vis, pt, 3, (0,255,255), -1)

    vis_path = os.path.join(SC_OUT_DIR, "trajectory.png")
    cv2.imwrite(vis_path, vis)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    lines = [",".join(results[0].keys())]
    for r in results[:10]:
        lines.append(",".join(str(v) for v in r.values()))
    if len(results) > 10:
        lines.append(f"... 共 {len(results)} 行")
    preview = "\n".join(lines)

    return vis_rgb, preview, f"分析完成! {len(results)} 个采样点\nCSV: {csv_path}\n轨迹: {vis_path}"

def sc_reset():
    return None, None, None, [], [], "已重置"

def build_app():
    video_dirs = list_video_dirs()

    with gr.Blocks(
        title="拒绝无效加班！！！(模板补全版)",
    ) as app:
        gr.Markdown("# SAM2 - YOLO → SHARED CONTROL 端到端 (模板补全版)")
        gr.Markdown(
            "视频切帧 → 点击选目标 → SAM分割传播 → **2D模板对齐补全遮挡** → "
            "导出 YOLO 数据集 → SHARED CONTROL 数据集"
        )

        with gr.Accordion("Step 0: 视频切帧", open=False):
            with gr.Row():
                video_file_dropdown = gr.Dropdown(
                    choices=list_video_files(), label="选择视频文件",
                    info="notebooks/videos/ 下的视频",
                )
                frame_interval = gr.Number(value=2, label="帧间隔", info="每N帧取1帧；论文协议为 every other frame", precision=0)
                extract_btn = gr.Button("切帧", variant="primary")
            extract_log = gr.Textbox(label="切帧结果", lines=2, interactive=False)

        with gr.Accordion("Step 1: SAM 选目标分割 + 模板补全, yolo_dataset 创建", open=True):

            points_state = gr.State([])
            labels_state = gr.State([])

            with gr.Row():

                with gr.Column(scale=1):
                    video_dir_dropdown = gr.Dropdown(
                        choices=video_dirs,
                        label="选择视频帧目录",
                        info="notebooks/video_to_img/ 下的子文件夹",
                    )
                    frame_slider = gr.Slider(
                        minimum=0, maximum=0, step=1, value=0,
                        label="帧索引",
                        info="选择要标注的帧 (建议选物体完全可见的帧)",
                    )
                    point_type = gr.Radio(
                        choices=["正样本 (前景)", "负样本 (背景)"],
                        value="正样本 (前景)",
                        label="点击类型",
                        info="正样本=目标区域，负样本=排除区域",
                    )
                    class_id = gr.Number(value=0, label="CLASS_ID", info="YOLO 类别 ID", precision=0)
                    info_box = gr.Textbox(label="信息", lines=6, interactive=False)

                    with gr.Row():
                        clear_btn = gr.Button("清除所有点", variant="secondary", size="sm")
                        undo_btn = gr.Button("撤销上一个点", variant="secondary", size="sm")

                    with gr.Accordion("🔧 模板补全 (遮挡修复)", open=True):
                        gr.Markdown(
                            "**原理**: 当物体被遮挡导致 SAM mask 不完整时，"
                            "用已知的2D完整轮廓模板对齐到当前位置，补全缺失区域。\n\n"
                            "**使用步骤**: 1) 提供物体2D模板图 → 2) 预览轮廓提取 → "
                            "3) 导出时自动补全"
                        )
                        enable_completion = gr.Checkbox(
                            value=False, label="启用模板补全",
                            info="勾选后导出时自动对遮挡帧进行补全"
                        )
                        template_path_input = gr.Textbox(
                            value=DEFAULT_TEMPLATE_PATH,
                            label="模板图片路径",
                            info="物体的完整2D轮廓图 (如 chain-direct.JPG)",
                            lines=1,
                        )
                        completion_thresh = gr.Slider(
                            minimum=0.3, maximum=0.95, value=0.7, step=0.05,
                            label="完整度阈值",
                            info="mask面积/参考帧面积 低于此值时触发补全 (0.7=面积少于70%就补全)"
                        )
                        angle_smooth_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                            label="角度平滑系数",
                            info="0=完全用历史角度, 1=完全用当前帧角度 (遮挡严重时建议低值)"
                        )
                        with gr.Row():
                            template_preview_btn = gr.Button(
                                "预览模板轮廓", variant="secondary", size="sm"
                            )
                        template_preview_img = gr.Image(
                            label="模板轮廓预览 (左=原图+轮廓, 右=二值mask)",
                            type="pil", interactive=False,
                        )
                        template_preview_info = gr.Textbox(
                            label="模板信息", lines=3, interactive=False
                        )

                with gr.Column(scale=2):
                    image_display = gr.Image(
                        label="点击图片选择目标点（绿色=正样本，红色=负样本）",
                        type="pil",
                        interactive=False,
                    )

            with gr.Row():
                export_btn = gr.Button(
                    "propagate → (模板补全) → 导出 YOLO 实例分割数据集",
                    variant="primary", size="lg"
                )

            with gr.Row():
                preview_image = gr.Image(label="Mask 预览", type="pil", interactive=False)

            export_log = gr.Textbox(label="logs", lines=20, interactive=False)

        with gr.Accordion("Step 2: Shared Control Dataset (迷宫配准 + 运动分析)", open=False):
            sc_maze_pts = gr.State([])
            sc_cam_pts = gr.State([])

            with gr.Row():
                sc_dataset_dd = gr.Dropdown(
                    choices=list_yolo_datasets(), label="选择 YOLO 数据集",
                    info="Step 1 导出的数据集 (images_vis + labels)",
                )
                sc_load_btn = gr.Button("加载数据", variant="primary")

            gr.Markdown(f"迷宫图固定路径: `{os.path.join(SC_DIR, 'maze1.png')}`")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 迷宫图 (点击 4 个特征点)")
                    sc_maze_img = gr.Image(label="迷宫", interactive=False)
                    sc_maze_info = gr.Textbox(label="迷宫特征点", value="(0/4)")
                with gr.Column():
                    gr.Markdown("#### 相机图 (点击对应 4 个点)")
                    sc_cam_img = gr.Image(label="相机", interactive=False)
                    sc_cam_info = gr.Textbox(label="相机特征点", value="(0/4)")

            with gr.Row():
                sc_alpha = gr.Slider(10, 100, value=70, step=5, label="叠加透明度 %")
                sc_step = gr.Number(value=4, label="采样间隔 (帧)", precision=0)
                sc_reg_btn = gr.Button("配准", variant="primary")
                sc_analyze_btn = gr.Button("运动分析", variant="primary")
                sc_reset_btn = gr.Button("重置")

            with gr.Row():
                sc_result_img = gr.Image(label="配准/轨迹结果")
            sc_csv_preview = gr.Textbox(label="CSV 预览", lines=8, interactive=False)
            sc_info = gr.Textbox(label="信息", lines=3, interactive=False)

            sc_load_btn.click(
                sc_on_dataset_change, [sc_dataset_dd],
                [sc_maze_img, sc_cam_img, sc_result_img, sc_info],
            )
            sc_maze_img.select(
                sc_click_maze, [sc_maze_pts],
                [sc_maze_img, sc_maze_pts, sc_maze_info],
            )
            sc_cam_img.select(
                sc_click_cam, [sc_cam_pts],
                [sc_cam_img, sc_cam_pts, sc_cam_info],
            )
            sc_reg_btn.click(
                sc_register, [sc_maze_pts, sc_cam_pts, sc_alpha],
                [sc_result_img, sc_info],
            )
            sc_analyze_btn.click(
                sc_analyze, [sc_step],
                [sc_result_img, sc_csv_preview, sc_info],
            )
            sc_reset_btn.click(
                sc_reset, [],
                [sc_maze_img, sc_cam_img, sc_result_img, sc_maze_pts, sc_cam_pts, sc_info],
            )

        extract_btn.click(
            fn=on_extract_frames,
            inputs=[video_file_dropdown, frame_interval],
            outputs=[extract_log, video_dir_dropdown],
        )

        video_dir_dropdown.change(
            fn=on_video_dir_change,
            inputs=[video_dir_dropdown],
            outputs=[image_display, frame_slider, frame_slider, info_box, points_state, labels_state, preview_image],
        )

        frame_slider.release(
            fn=on_frame_change,
            inputs=[video_dir_dropdown, frame_slider, points_state, labels_state],
            outputs=[image_display],
        )

        image_display.select(
            fn=on_image_click,
            inputs=[video_dir_dropdown, frame_slider, point_type, points_state, labels_state],
            outputs=[image_display, points_state, labels_state, info_box],
        )

        clear_btn.click(
            fn=on_clear_points,
            inputs=[video_dir_dropdown, frame_slider],
            outputs=[image_display, points_state, labels_state, info_box],
        )

        undo_btn.click(
            fn=on_undo_point,
            inputs=[video_dir_dropdown, frame_slider, points_state, labels_state],
            outputs=[image_display, points_state, labels_state, info_box],
        )

        template_preview_btn.click(
            fn=preview_template_extraction,
            inputs=[template_path_input],
            outputs=[template_preview_img, template_preview_info],
        )

        def export_and_refresh(vdir, fidx, pts, lbls, cid,
                               en_comp, tpl_path, comp_th, ang_sm):
            log = on_export_yolo(vdir, fidx, pts, lbls, cid,
                                 en_comp, tpl_path, comp_th, ang_sm)
            return log, gr.update(choices=list_yolo_datasets())

        export_btn.click(
            fn=export_and_refresh,
            inputs=[video_dir_dropdown, frame_slider, points_state, labels_state, class_id,
                    enable_completion, template_path_input, completion_thresh, angle_smooth_slider],
            outputs=[export_log, sc_dataset_dd],
        )

    return app

if __name__ == '__main__':
    app = build_app()
    app.launch(
        server_name='0.0.0.0',
        server_port=7861,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
