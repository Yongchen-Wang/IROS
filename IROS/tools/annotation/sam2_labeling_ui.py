#!/usr/bin/env python3

import os
import sys
import shutil
import subprocess
import time
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

import pandas as pd
import glob
import json
from pathlib import Path

from sam2.build_sam import build_sam2_video_predictor
from tools.annotation import template_alignment_ui as ta
from tools.annotation import maze_processing as mp
from iros.paths import (
    CHECKPOINT_ROOT,
    DATA_ROOT,
    FRAME_ROOT,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    RAW_DATA_ROOT,
    VIDEO_ROOT,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = str(FRAME_ROOT)
VIDEOS_DIR = str(VIDEO_ROOT)
RAW_DATA_DIR = str(RAW_DATA_ROOT)
YOLO_DATASET_DIR = str(OUTPUT_ROOT / "yolo_datasets")
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

def list_video_files():

    if not os.path.isdir(VIDEOS_DIR):
        return []
    exts = ('.mp4', '.avi', '.mov', '.mkv')
    return sorted([f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(exts)])

def list_raw_data_dirs():

    if not os.path.isdir(RAW_DATA_DIR):
        return []
    dirs = []
    for d in sorted(os.listdir(RAW_DATA_DIR)):
        full = os.path.join(RAW_DATA_DIR, d)
        if os.path.isdir(full):
            files = os.listdir(full)
            has_mp4 = any(f.lower().endswith('.mp4') for f in files)
            has_csv = any(f.lower().endswith('.csv') for f in files)
            if has_mp4 and has_csv:
                dirs.append(d)
    return dirs

def on_process_raw_data(dataset_name, frame_interval):

    if not dataset_name:
        return "请先选择数据文件夹"

    dataset_path = os.path.join(RAW_DATA_DIR, dataset_name)
    if not os.path.isdir(dataset_path):
        return f"文件夹不存在: {dataset_path}"

    files = os.listdir(dataset_path)
    mp4_files = [f for f in files if f.lower().endswith('.mp4')]
    csv_files = [f for f in files if f.lower().endswith('.csv')]

    if not mp4_files:
        return f"未找到视频文件: {dataset_path}"
    if not csv_files:
        return f"未找到 CSV 文件: {dataset_path}"

    video_file = mp4_files[0]
    csv_file = csv_files[0]

    video_path = os.path.join(dataset_path, video_file)
    csv_path = os.path.join(dataset_path, csv_file)

    output_name = Path(dataset_name).stem
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
    except Exception as e:
        return f"切帧失败: {e}"

    frame_files = sorted([f for f in os.listdir(output_path) if f.lower().endswith('.jpg')])
    n_frames = len(frame_files)

    if n_frames == 0:
        return "切帧失败: 未生成任何帧"

    try:
        df = pd.read_csv(csv_path)

        has_raw = 'fsr_L_raw' in df.columns and 'fsr_R_raw' in df.columns
        has_processed = 'fsr_L' in df.columns and 'fsr_R' in df.columns
        if not has_raw and not has_processed:
            return f"CSV 文件缺少 FSR 列 (fsr_L_raw/fsr_L): {csv_path}"

        fsr_indices = list(range(0, len(df), frame_interval))
        if has_raw:
            fsr_L_raw_series = df['fsr_L_raw'].iloc[fsr_indices].reset_index(drop=True)
            fsr_R_raw_series = df['fsr_R_raw'].iloc[fsr_indices].reset_index(drop=True)
        elif has_processed:
            fsr_L_raw_series = df['fsr_L'].iloc[fsr_indices].reset_index(drop=True)
            fsr_R_raw_series = df['fsr_R'].iloc[fsr_indices].reset_index(drop=True)
        else:
            fsr_L_raw_series = pd.Series([0.0] * len(fsr_indices))
            fsr_R_raw_series = pd.Series([0.0] * len(fsr_indices))

        def _pad_or_trunc(series, target_len):
            if len(series) > target_len:
                return series[:target_len]
            elif len(series) < target_len:
                last_val = series.iloc[-1] if len(series) > 0 else 0
                pad = pd.Series([last_val] * (target_len - len(series)))
                return pd.concat([series, pad], ignore_index=True)
            return series

        fsr_L_raw_series = _pad_or_trunc(fsr_L_raw_series, n_frames)
        fsr_R_raw_series = _pad_or_trunc(fsr_R_raw_series, n_frames)

        fsr_df = pd.DataFrame({
            'frame_idx': range(n_frames),
            'fsr_L_raw': fsr_L_raw_series.values,
            'fsr_R_raw': fsr_R_raw_series.values,
        })
        fsr_csv_path = os.path.join(output_path, 'fsr_data.csv')
        fsr_df.to_csv(fsr_csv_path, index=False)

    except Exception as e:
        return f"处理 CSV 失败: {e}"

    return (f"处理完成: {dataset_name}\n"
            f"- 切帧: {n_frames} 帧 (每 {frame_interval} 帧取 1 帧)\n"
            f"- FSR 数据: {len(fsr_L_raw_series)} 条记录\n"
            f"- 保存至: {output_path}\n"
            f"- FSR 数据: {fsr_csv_path}")

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

def draw_mask_overlay(pil_img, mask, alpha=0.45):

    img_np = np.array(pil_img).copy()
    mask_2d = np.squeeze(mask)
    if mask_2d.ndim != 2:
        return pil_img

    h, w = img_np.shape[:2]
    mh, mw = mask_2d.shape
    if mh != h or mw != w:
        mask_2d = cv2.resize(mask_2d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = img_np.copy()
    overlay[mask_2d > 0] = [255, 50, 50]
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
        inference_state = predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

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

def on_export_yolo(video_dir_name, frame_idx, points_state, labels_state, class_id, progress=gr.Progress()):

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

        progress(0.0, desc="初始化 SAM2 ...")
        inference_state = predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

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
        log(f"   输出目录: {output_dir}")
        log(f"     ├── images/     (原图)")
        log(f"     ├── labels/     (YOLO 实例分割label)")
        log(f"     └── images_vis/ (可视化)")

        return "\n".join(log_lines)

    except Exception as e:
        traceback.print_exc()
        return f"导出失败: {str(e)}\n{traceback.format_exc()}"

def build_app():
    video_dirs = list_video_dirs()

    with gr.Blocks(
        title="噜噜酱真棒！",
    ) as app:
        gr.Markdown("# SAM2 - YOLO → SHARED CONTREOL")
        gr.Markdown("视频切帧 → 点击选目标 → sam分割传播 → 导出 YOLO 数据集 → SHARED CONTREOL 数据集")

        with gr.Accordion("Step 0: 视频切帧", open=False):
            with gr.Row():
                video_file_dropdown = gr.Dropdown(
                    choices=list_video_files(), label="选择视频文件",
                    info="notebooks/videos/ 下的视频",
                )
                frame_interval = gr.Number(value=2, label="帧间隔", info="每N帧取1帧；论文协议为 every other frame", precision=0)
                extract_btn = gr.Button("切帧", variant="primary")
            extract_log = gr.Textbox(label="切帧结果", lines=2, interactive=False)

        with gr.Accordion("Step 1: SAM 选目标分割, yolo_dataset 创建", open=False):

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
                        info="选择要标注的帧",
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

                with gr.Column(scale=2):
                    image_display = gr.Image(
                        label="点击图片选择目标点（绿色=正样本，红色=负样本）",
                        type="pil",
                        interactive=False,
                    )

            with gr.Row():
                export_btn = gr.Button("propagate & 导出 YOLO 实例分割数据集", variant="primary", size="lg")

            with gr.Row():
                preview_image = gr.Image(label="Mask 预览", type="pil", interactive=False)

            export_log = gr.Textbox(label="logs", lines=15, interactive=False)

        with gr.Accordion("Step 2: Shared Control Dataset (mask align + 特征导入)", open=False):

            with gr.Accordion("Step 2-0: 原始数据处理 (raw_data 导入)", open=False):
                gr.Markdown("""
                **功能说明：**
                - 从 `raw_data/` 文件夹读取视频和 CSV 数据
                - 自动切帧（论文协议：every other frame，即帧间隔默认为2）
                - 提取 CSV 中的 FSR 数据（fsr_L_raw, fsr_R_raw）
                - 保存到 `video_to_img/` 目录供后续步骤使用
                """)
                with gr.Row():
                    t40_dataset_dd = gr.Dropdown(
                        choices=list_raw_data_dirs(), label="选择原始数据文件夹",
                        info="raw_data/ 下的子文件夹",
                    )
                    t40_frame_interval = gr.Number(value=2, label="帧间隔", info="每N帧取1帧；论文协议默认2", precision=0)
                    t40_process_btn = gr.Button("处理数据", variant="primary")
                t40_result = gr.Textbox(label="处理结果", lines=5, interactive=False)

            with gr.Accordion("Step 2-1: mask align", open=False):
                with gr.Row():
                    t4_video_dir_dd = gr.Dropdown(
                        choices=ta.list_video_dirs(),
                        label="选择视频帧目录",
                        info="video_to_img/ 下的子文件夹",
                    )
                    t4_template_path = gr.Textbox(
                        value=ta.DEFAULT_TEMPLATE, label="mask路径"
                    )
                    t4_load_btn = gr.Button("加载数据", variant="primary")

                t4_pos_x_state = gr.State(None)
                t4_pos_y_state = gr.State(None)

                with gr.Row():
                    with gr.Column(scale=1):
                        t4_tpl_img = gr.Image(label="mask轮廓", interactive=False, height=220)
                        gr.Markdown("点击右图定位，滑条调缩放/旋转/膨胀")
                        t4_scale = gr.Slider(0.01, 0.5, 0.075, step=0.005, label="缩放")
                        t4_angle = gr.Slider(-180, 180, -92.5, step=0.5, label="旋转 (度)")
                        t4_dilate = gr.Slider(0, 10, 3, step=1, label="mask膨胀 (px)")
                        t4_alpha = gr.Slider(10, 80, 40, step=5, label="透明度 %")
                        with gr.Row():
                            t4_update_btn = gr.Button("刷新", variant="secondary")
                            t4_save_btn = gr.Button("保存对齐", variant="primary")
                        t4_info_box = gr.Textbox(label="状态", lines=3, interactive=False)
                    with gr.Column(scale=2):
                        t4_preview_img = gr.Image(
                            label="点击 | 绿轮廓 | mask | 红十字中心",
                            interactive=False,
                        )
                        t4_preview_info = gr.Textbox(label="预览", lines=1, interactive=False)
                t4_save_result = gr.Textbox(label="保存结果", lines=3, interactive=False)

            with gr.Accordion("Step 2-2: SAM传播 + mask align", open=False):
                t4_pts_state = gr.State([])
                t4_labels_state = gr.State([])

                with gr.Row():
                    with gr.Column(scale=1):
                        t4_frame_slider = gr.Slider(0, 1000, 0, step=1, label="标注帧索引")
                        t4_point_type = gr.Radio(
                            ["正样本 (前景)", "负样本 (背景)"],
                            value="正样本 (前景)",
                            label="点击类型",
                        )
                        t4_class_id = gr.Number(value=0, label="CLASS_ID", precision=0)
                        t4_track_angle = gr.Checkbox(value=True, label="角度追踪")
                        t4_use_endpoints = gr.Checkbox(
                            value=True, label="端点圆匹配", info="默认开启"
                        )
                        with gr.Row():
                            t4_clear_btn = gr.Button("清除标注", variant="secondary")
                            t4_preview_btn = gr.Button("预览 SAM Mask", variant="secondary")
                        t4_run_btn = gr.Button(
                            "传播 + 补全 + 数据集导出(SAM + MASK ALIGN)", variant="primary", size="lg"
                        )
                        t4_pts_info = gr.Textbox(label="标注", lines=2, interactive=False)
                    with gr.Column(scale=2):
                        t4_annotate_img = gr.Image(
                            label="点击标注目标 (绿=前景, 红=背景)", interactive=False
                        )

                with gr.Row():
                    t4_sam_preview_img = gr.Image(label="SAM mask vs 模板", interactive=False)
                    t4_sam_preview_info = gr.Textbox(label="对比信息", lines=12, interactive=False)

                t4_log = gr.Textbox(label="运行日志", lines=12, interactive=False)

                with gr.Row():
                    t4_browse_slider = gr.Slider(0, 1000, 0, step=1, label="浏览帧索引")
                with gr.Row():
                    t4_browse_img = gr.Image(label="左: SAM分割 + mask align (images_vis)", interactive=False)
                    t4_browse_img_tpl_only = gr.Image(
                        label="右: 仅mask (images_vis_template_only)", interactive=False
                    )
                t4_browse_info = gr.Textbox(label="帧信息", lines=1, interactive=False)

            with gr.Accordion("Step 2-3: 迷宫特征生成", open=False):
                gr.Markdown(
                    "**配准参数已固定** (cx=624, cy=578, scale=0.565, angle=0°)\n\n"
                    "选择数据集 → 选择活跃目标 → 点击「生成迷宫特征」→ 生成 `maze_features.pkl`"
                )

                _PRESET_TARGETS = {
                    'A': [800, 580],
                    'B': [480, 800],
                    'C': [130, 720],
                }
                t43_targets_state = gr.State(_PRESET_TARGETS)

                with gr.Row():
                    t43_dataset_dd = gr.Dropdown(
                        choices=mp.list_mask_align_datasets(),
                        label="选择 mask align 数据集",
                        info="mask_align_sam2_dataset/ 下的文件夹",
                    )
                    t43_target_selector = gr.Radio(
                        choices=["A", "B", "C"],
                        value="A",
                        label="活跃目标",
                        info="A=(800,580)  B=(480,800)  C=(130,720)",
                    )
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("骨架算法高级参数 (一般无需修改)", open=False):
                            t43_bif_radius = gr.Number(
                                value=25, label="分叉聚类半径 (px)", precision=0,
                        )
                            t43_boundary_margin = gr.Number(
                                value=30, label="边界端点排除 (px)", precision=0,
                        )
                            t43_curv_arc = gr.Number(
                                value=30, label="曲率弧长窗口 (px)", precision=0,
                        )
                            t43_curv_sigma = gr.Number(
                                value=3, label="曲率平滑 sigma",
                            )
                with gr.Row():
                    t43_gen_features_btn = gr.Button(
                        "生成迷宫特征 (maze_features.pkl)",
                        variant="primary", size="lg",
                    )
                t43_gen_features_result = gr.Textbox(
                    label="生成结果", lines=15, interactive=False,
                        )

        with gr.Accordion("Step 3: Alpha 回归训练 (Bi-CAST 与基线)", open=False):
            gr.Markdown("""
            **功能说明：**
            - 训练 Bi-CAST 与论文对比基线, 回归双臂控制权重 α
            - 输入: 连续 4 帧图像 + 双臂 FSR 特征 + 安全特征 [IoU, d_wall]
            - 输出: 左/右臂 α ∈ [0, 0.9] (chunk = 5 步)
            """)

            with gr.Accordion("Step 3-1: 数据集检查", open=False):
                gr.Markdown(
                    "训练数据集 = `images/` + `alpha_labels.pkl`。\n"
                    "标签由三名标注者独立打分后取平均: "
                    "`python -m tools.annotation.merge_alpha_annotations "
                    "--annotations a.csv b.csv c.csv --recording <dataset>`; "
                    "FSR / IoU / d_wall 由 "
                    "`tools/data/import_wall_distance_fsr.py` 导入。"
                )
                with gr.Row():
                    t4_dataset_dd = gr.Dropdown(
                        choices=mp.list_mask_align_datasets(),
                        label="选择数据集",
                        info="包含 images/ 与 alpha_labels.pkl 的目录",
                    )
                with gr.Row():
                    t4_prepare_btn = gr.Button("检查数据集", variant="primary")
                    t4_prepare_result = gr.Textbox(label="检查结果", lines=5, interactive=False)

            with gr.Accordion("Step 3-2: 网络选择与训练", open=False):
                gr.Markdown("### 网络架构选择")
                t4_network_dd = gr.Dropdown(
                    choices=[
                        "three_layer_cnn", "resnet18", "resnet50",
                        "tsm_resnet18", "r2plus1d", "mvit", "vit", "bicast"
                    ],
                    value=["bicast"],
                    label="选择网络架构 (可多选)",
                    info="按住 Ctrl/Cmd 可多选",
                    multiselect=True
                )

                gr.Markdown("### 训练参数")
                with gr.Row():
                    with gr.Column():
                        t4_epochs = gr.Number(value=50, label="Epochs", precision=0)
                        t4_batch_size = gr.Number(value=8, label="Batch Size", precision=0)
                        t4_lr = gr.Number(value=1e-4, label="学习率")
                    with gr.Column():
                        t4_weight_decay = gr.Number(value=1e-4, label="Weight Decay")
                        t4_val_ratio = gr.Number(value=0.2, label="验证集比例")
                        t4_test_ratio = gr.Number(value=0.2, label="测试集比例")
                        t4_img_size = gr.Number(value=224, label="图像尺寸", precision=0)

                gr.Markdown("### LSTM/GRU/TCN/Transformer 额外参数")
                with gr.Row():
                    t4_hidden_dim = gr.Number(value=128, label="Hidden Dim", precision=0)
                    t4_num_layers = gr.Number(value=1, label="Num Layers", precision=0)

                with gr.Row():
                    t4_train_btn = gr.Button("开始训练", variant="primary", size="lg")
                    t4_stop_btn = gr.Button("停止训练", variant="stop")

                t4_train_log = gr.Textbox(label="训练日志", lines=15, interactive=False)
                t4_train_result = gr.Textbox(label="训练结果", lines=10, interactive=False)

        def extract_and_refresh(video_file, interval):
            log_text, vd_update = on_extract_frames(video_file, interval)
            return log_text, vd_update, vd_update

        extract_btn.click(
            fn=extract_and_refresh,
            inputs=[video_file_dropdown, frame_interval],
            outputs=[extract_log, video_dir_dropdown, t4_video_dir_dd],
        )

        def process_raw_data_and_refresh(dataset_name, frame_interval):
            result = on_process_raw_data(dataset_name, frame_interval)
            new_dirs = list_video_dirs()
            return (
                result,
                gr.update(choices=new_dirs),
                gr.update(choices=new_dirs),
            )

        t40_process_btn.click(
            fn=process_raw_data_and_refresh,
            inputs=[t40_dataset_dd, t40_frame_interval],
            outputs=[t40_result, video_dir_dropdown, t4_video_dir_dd],
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

        export_btn.click(
            fn=on_export_yolo,
            inputs=[video_dir_dropdown, frame_slider, points_state, labels_state, class_id],
            outputs=[export_log],
        )

        t4_load_btn.click(
            fn=ta.on_load_data,
            inputs=[t4_video_dir_dd, t4_template_path],
            outputs=[t4_tpl_img, t4_preview_img, t4_pos_x_state, t4_pos_y_state, t4_scale, t4_info_box],
        )
        t4_preview_img.select(
            fn=ta.on_click_frame,
            inputs=[t4_pos_x_state, t4_pos_y_state, t4_video_dir_dd, t4_template_path,
                    t4_scale, t4_angle, t4_dilate, t4_alpha],
            outputs=[t4_preview_img, t4_pos_x_state, t4_pos_y_state, t4_preview_info],
        )
        _t4_s_in = [t4_pos_x_state, t4_pos_y_state, t4_video_dir_dd, t4_template_path,
                    t4_scale, t4_angle, t4_dilate, t4_alpha]
        _t4_s_out = [t4_preview_img, t4_preview_info]
        for ctrl in [t4_scale, t4_angle, t4_dilate, t4_alpha]:
            ctrl.release(fn=ta.on_slider_change, inputs=_t4_s_in, outputs=_t4_s_out)
        t4_update_btn.click(fn=ta.on_slider_change, inputs=_t4_s_in, outputs=_t4_s_out)
        t4_save_btn.click(
            fn=ta.on_save_params,
            inputs=[t4_pos_x_state, t4_pos_y_state, t4_video_dir_dd, t4_template_path, t4_scale, t4_angle],
            outputs=[t4_save_result],
        )

        def t4_on_dir_or_frame(vdir, fi):
            img, info = ta.on_s2_load_frame(vdir, fi)
            frames = ta.get_sorted_frames(vdir) if vdir else []
            return (
                img, info,
                gr.update(maximum=max(len(frames) - 1, 0)),
                gr.update(maximum=max(len(frames) - 1, 0)),
                [], [],
            )

        t4_video_dir_dd.change(
            fn=t4_on_dir_or_frame,
            inputs=[t4_video_dir_dd, t4_frame_slider],
            outputs=[t4_annotate_img, t4_pts_info, t4_frame_slider, t4_browse_slider, t4_pts_state, t4_labels_state],
        )
        t4_frame_slider.release(
            fn=lambda vd, fi: ta.on_s2_load_frame(vd, fi),
            inputs=[t4_video_dir_dd, t4_frame_slider],
            outputs=[t4_annotate_img, t4_pts_info],
        )
        t4_annotate_img.select(
            fn=ta.on_s2_click,
            inputs=[t4_video_dir_dd, t4_frame_slider, t4_point_type, t4_pts_state, t4_labels_state],
            outputs=[t4_annotate_img, t4_pts_state, t4_labels_state, t4_pts_info],
        )
        t4_clear_btn.click(
            fn=ta.on_s2_clear,
            inputs=[t4_video_dir_dd, t4_frame_slider],
            outputs=[t4_annotate_img, t4_pts_state, t4_labels_state, t4_pts_info],
        )
        t4_preview_btn.click(
            fn=ta.on_s2_preview_mask,
            inputs=[t4_video_dir_dd, t4_template_path, t4_frame_slider,
                    t4_pts_state, t4_labels_state, t4_pos_x_state, t4_pos_y_state, t4_scale, t4_angle],
            outputs=[t4_sam_preview_img, t4_sam_preview_info],
        )
        def propagate_and_refresh_downstream(*args):
            log_text, browse_img = ta.on_s2_propagate_and_complete(*args)
            new_maze_ds = mp.list_mask_align_datasets()
            return (
                log_text, browse_img,
                gr.update(choices=new_maze_ds),
                gr.update(choices=new_maze_ds),
            )

        t4_run_btn.click(
            fn=propagate_and_refresh_downstream,
            inputs=[t4_video_dir_dd, t4_template_path, t4_frame_slider,
                    t4_pts_state, t4_labels_state, t4_class_id,
                    t4_pos_x_state, t4_pos_y_state, t4_scale, t4_angle, t4_dilate,
                    t4_track_angle, t4_use_endpoints],
            outputs=[t4_log, t4_browse_img,
                     t43_dataset_dd, t4_dataset_dd],
        )
        def t4_on_browse_dual(vdir, fi):
            fi = int(fi)
            img_name = f"{fi:05d}.jpg"
            left = None
            right = None

            if vdir:
                out_dir = os.path.join(ta.SAM2_DATASET_DIR, vdir)
                left_path = os.path.join(out_dir, "images_vis", img_name)
                right_path = os.path.join(out_dir, "images_vis_template_only", img_name)
                if os.path.exists(left_path):
                    left = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)
                if os.path.exists(right_path):
                    right = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)

            info = f"帧 {fi}"
            if left is None:
                left_mem, info_mem = ta.on_s2_browse(vdir, fi)
                left = left_mem
                info = info_mem
            else:
                if right is None:
                    info = f"帧 {fi} | 找不到 images_vis_template_only/{img_name}"
                else:
                    info = f"帧 {fi} | 左=images_vis | 右=images_vis_template_only"
            return left, right, info

        t4_browse_slider.release(
            fn=t4_on_browse_dual,
            inputs=[t4_video_dir_dd, t4_browse_slider],
            outputs=[t4_browse_img, t4_browse_img_tpl_only, t4_browse_info],
        )

        def gen_maze_features_and_refresh(
            dataset_name, targets_dict, target_selector,
            bif_radius, boundary_margin, curv_arc, curv_sigma,
        ):
            result = mp.generate_maze_features(
                dataset_name, targets_dict, target_selector,
                bif_cluster_radius=bif_radius,
                boundary_margin=boundary_margin,
                curvature_arc_length=curv_arc,
                curvature_smooth_sigma=curv_sigma,
            )
            return result

        t43_gen_features_btn.click(
            gen_maze_features_and_refresh,
            [t43_dataset_dd,
             t43_targets_state, t43_target_selector,
             t43_bif_radius, t43_boundary_margin,
             t43_curv_arc, t43_curv_sigma],
            [t43_gen_features_result],
        )

        def on_check_regression_dataset(dataset_name):

            if not dataset_name:
                return "请先选择数据集"
            root = DATA_ROOT / dataset_name
            img_dir = root / "images"
            lbl = root / "alpha_labels.pkl"
            lines = [f"数据集: {root}"]
            n_img = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png"))) if img_dir.is_dir() else 0
            lines.append(f"images/: {'✅ ' + str(n_img) + ' 帧' if n_img else '❌ 缺失或为空'}")
            if lbl.exists():
                import pickle as _pkl
                with open(lbl, "rb") as f:
                    lab = _pkl.load(f)
                need = ["alpha_L", "alpha_R", "fsr_L", "fsr_R", "iou", "d_wall"]
                for k in need:
                    lines.append(f"alpha_labels.pkl[{k}]: "
                                 + (f"✅ {len(lab[k])}" if k in lab else "❌ 缺失"))
                missing = [k for k in need if k not in lab]
                if missing:
                    lines.append("提示: alpha_L/alpha_R 用 merge_alpha_annotations.py 合并; "
                                 "fsr/iou/d_wall 用 tools/data/import_wall_distance_fsr.py 导入")
            else:
                lines.append("alpha_labels.pkl: ❌ 缺失")
            return "\n".join(lines)

        t4_prepare_btn.click(
            on_check_regression_dataset,
            [t4_dataset_dd],
            [t4_prepare_result],
        )

        training_state = gr.State({"running": False, "process": None})

        def on_start_training(
            dataset_name, networks, 
            epochs, batch_size, lr, weight_decay, val_ratio, test_ratio, img_size,
            hidden_dim, num_layers
        ):

            if not dataset_name:
                yield "请先选择数据集", "No dataset selected"
                return

            if isinstance(networks, str):
                networks = [networks]

            if not networks:
                yield "请先选择网络", "No network selected"
                return

            data_root = str(DATA_ROOT / dataset_name)
            ann_path = os.path.join(data_root, "alpha_labels.pkl")
            if not os.path.exists(ann_path):
                yield f"数据集不存在: {ann_path}\n请先合并标注 (tools/annotation/merge_alpha_annotations.py) 并导入 FSR/d_wall (tools/data/import_wall_distance_fsr.py)", "Dataset not found"
                return

            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node=1",
                str(PROJECT_ROOT / "training" / "core" / "train_model.py"),
                "--data_root", data_root,
                "--epochs", str(int(epochs)),
                "--batch_size", str(int(batch_size)),
                "--lr", str(lr),
                "--wd", str(weight_decay),
                "--val_ratio", str(val_ratio),
                "--test_ratio", str(test_ratio),
                "--img_size", str(int(img_size)),
            ]

            cmd.extend(["--models"] + list(networks))

            log_lines = [f"开始训练 {len(networks)} 个网络: {', '.join(networks)}..."]
            log_lines.append(f"命令: {' '.join(cmd)}")
            log_lines.append("-" * 50)

            yield "\n".join(log_lines), "Training started..."

            try:

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(PROJECT_ROOT)
                )

                import select
                epoch_count = 0
                while True:

                    ret = process.poll()
                    if ret is not None:
                        break

                    ready, _, _ = select.select([process.stdout], [], [], 0.5)
                    if ready:
                        line = process.stdout.readline()
                        if line:
                            log_lines.append(line.strip())

                            if len(log_lines) > 100:
                                log_lines = log_lines[-50:]

                            if "Epoch" in line or "epoch" in line:
                                epoch_count += 1

                            yield "\n".join(log_lines), f"Training... Epoch {epoch_count}"

                    time.sleep(0.1)

                remaining = process.stdout.read()
                if remaining:
                    log_lines.extend(remaining.strip().split('\n'))

                log_lines.append("-" * 50)
                log_lines.append(f"训练完成! 进程返回码: {ret}")

                save_dir = str(OUTPUT_ROOT / "authority_models")
                if os.path.exists(save_dir):
                    model_files = sorted(Path(save_dir).glob(f"**/model_*.pth"))
                    if model_files:
                        log_lines.append(f"最佳模型: {model_files[-1]}")

                yield "\n".join(log_lines), "Training completed"

            except Exception as e:
                import traceback
                yield "\n".join(log_lines) + f"\n错误: {e}\n{traceback.format_exc()}", "Training failed"

        t4_train_btn.click(
            on_start_training,
            [t4_dataset_dd, t4_network_dd,
             t4_epochs, t4_batch_size, t4_lr, t4_weight_decay,
             t4_val_ratio, t4_test_ratio, t4_img_size,
             t4_hidden_dim, t4_num_layers],
            [t4_train_log, t4_train_result],
        )

    return app

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SAM2-YOLO Gradio UI')
    parser.add_argument('--port', type=int, default=7860, help='Server port (default: 7860)')
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name='0.0.0.0',
        server_port=args.port,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
