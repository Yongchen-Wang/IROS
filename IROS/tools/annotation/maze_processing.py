#!/usr/bin/env python3

import os
import sys
import json
import glob
import pickle
import numpy as np
import cv2
import gradio as gr

try:
    from iros.maze import (
        MazeFeatureExtractor, load_and_preprocess_maze
    )
    MAZE_EXTRACTOR_AVAILABLE = True
except Exception as _e:
    MAZE_EXTRACTOR_AVAILABLE = False
    import traceback
    print(f"Warning: maze feature extractor is not available: {_e}")
    traceback.print_exc()

from iros.paths import ASSET_ROOT, DATA_ROOT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAZE_PATH = str(ASSET_ROOT / "maze.png")
MASK_ALIGN_DIR = str(DATA_ROOT)

def list_mask_align_datasets():

    if not os.path.isdir(MASK_ALIGN_DIR):
        return []
    return sorted([
        d for d in os.listdir(MASK_ALIGN_DIR)
        if os.path.isdir(os.path.join(MASK_ALIGN_DIR, d, 'images_vis_template_only'))
    ])

def _params_path(dataset_name):

    return os.path.join(MASK_ALIGN_DIR, dataset_name,
                        'registration_output', 'maze_params.json')

def _build_affine(mw, mh, cx, cy, scale, angle_deg):

    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    mx, my = mw / 2.0, mh / 2.0
    M = np.array([
        [scale * cos_t, -scale * sin_t, cx - scale * (cos_t * mx - sin_t * my)],
        [scale * sin_t,  scale * cos_t, cy - scale * (sin_t * mx + cos_t * my)],
    ], dtype=np.float64)
    return M

def _get_maze_channels():

    if not os.path.exists(MAZE_PATH):
        return None
    maze_raw = cv2.imread(MAZE_PATH, cv2.IMREAD_UNCHANGED)
    if maze_raw is None:
        return None
    if len(maze_raw.shape) == 3 and maze_raw.shape[2] == 4:
        bgr, a = maze_raw[:, :, :3], maze_raw[:, :, 3]
    else:
        bgr = maze_raw if len(maze_raw.shape) == 3 else cv2.cvtColor(maze_raw, cv2.COLOR_GRAY2BGR)
        a = np.full(bgr.shape[:2], 255, np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, a, gray

def _load_maze_rgb():

    if not os.path.exists(MAZE_PATH):
        return None
    m = cv2.imread(MAZE_PATH, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if len(m.shape) == 3 and m.shape[2] == 4:
        a = m[:, :, 3:4].astype(np.float32) / 255
        bgr = m[:, :, :3].astype(np.float32)
        rgb = (bgr * a + 255 * (1 - a)).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(m, cv2.COLOR_BGR2RGB)

def _overlay(cam_bgr, cx, cy, scale, angle_deg, alpha_pct):

    md = _get_maze_channels()
    if md is None:
        return cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
    bgr, a, _ = md
    mh, mw = bgr.shape[:2]
    h, w = cam_bgr.shape[:2]
    M = _build_affine(mw, mh, cx, cy, scale, angle_deg)
    w_alpha = cv2.warpAffine(a, M, (w, h))
    mask_f = (w_alpha.astype(np.float32) / 255.0)[..., None] * (alpha_pct / 100.0)

    overlay = np.zeros_like(cam_bgr)
    overlay[:] = (50, 80, 255)
    result = np.clip(
        cam_bgr.astype(np.float32) * (1 - mask_f) + overlay.astype(np.float32) * mask_f,
        0, 255,
    ).astype(np.uint8)

    cv2.drawMarker(result, (int(cx), int(cy)), (0, 0, 255),
                   cv2.MARKER_CROSS, 20, 2)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

def _first_cam(dataset_name):

    vis_dir = os.path.join(MASK_ALIGN_DIR, dataset_name, 'images_vis_template_only')
    imgs = sorted(glob.glob(os.path.join(vis_dir, '*.jpg')))
    if not imgs:
        return None, 0
    cam = cv2.imread(imgs[0])
    return cam, len(imgs)

def on_load(dataset_name):

    if not dataset_name:
        return None, None, None, None, gr.update(), gr.update(), "请选择数据集"

    cam_bgr, n_imgs = _first_cam(dataset_name)
    if cam_bgr is None:
        return None, None, None, None, gr.update(), gr.update(), \
            f"未找到图像: {os.path.join(MASK_ALIGN_DIR, dataset_name, 'images_vis_template_only')}"

    img_h, img_w = cam_bgr.shape[:2]
    cx, cy = img_w / 2.0, img_h / 2.0
    scale_val, angle_val = 0.5, 0.0

    pf = _params_path(dataset_name)
    loaded = False
    if os.path.exists(pf):
        try:
            with open(pf) as f:
                p = json.load(f)
            cx = p.get('cx', cx)
            cy = p.get('cy', cy)
            scale_val = p.get('scale', scale_val)
            angle_val = p.get('angle_deg', angle_val)
            loaded = True
        except Exception:
            pass

    overlay_rgb = _overlay(cam_bgr, cx, cy, scale_val, angle_val, 50)
    maze_rgb = _load_maze_rgb()

    info = f"已加载 {n_imgs} 帧, 尺寸 {img_w}×{img_h}"
    info += f"\n{'✅ 已加载参数: ' + pf if loaded else '(无已保存参数, 使用默认值)'}"
    if maze_rgb is not None:
        info += f"\n迷宫图: {MAZE_PATH}"
    else:
        info += f"\n⚠️ 迷宫图不存在: {MAZE_PATH}"

    return (
        maze_rgb, overlay_rgb,
        cx, cy,
        gr.update(value=scale_val),
        gr.update(value=angle_val),
        info,
    )

def on_click(cx_old, cy_old, dataset_name, scale, angle, alpha,
             evt: gr.SelectData):

    x, y = evt.index
    if not dataset_name:
        return None, cx_old, cy_old, "请先加载数据"
    cam_bgr, _ = _first_cam(dataset_name)
    if cam_bgr is None:
        return None, cx_old, cy_old, "未找到图像"
    ov = _overlay(cam_bgr, x, y, scale, angle, alpha)
    return ov, float(x), float(y), \
        f"center=({x}, {y}), scale={scale:.3f}, angle={angle:.1f}°"

def on_slider(cx, cy, dataset_name, scale, angle, alpha):

    if not dataset_name or cx is None or cy is None:
        return None, "请先加载数据并点击设置位置"
    cam_bgr, _ = _first_cam(dataset_name)
    if cam_bgr is None:
        return None, "未找到图像"
    ov = _overlay(cam_bgr, cx, cy, scale, angle, alpha)
    return ov, f"center=({cx:.0f}, {cy:.0f}), scale={scale:.3f}, angle={angle:.1f}°"

def save_params(dataset_name, cx, cy, scale, angle):

    if not dataset_name:
        return "请先选择数据集"
    if cx is None or cy is None:
        return "请先点击设置位置"
    out_dir = os.path.join(MASK_ALIGN_DIR, dataset_name, 'registration_output')
    os.makedirs(out_dir, exist_ok=True)
    params = dict(cx=float(cx), cy=float(cy),
                  scale=float(scale), angle_deg=float(angle),
                  maze_path=MAZE_PATH)
    pf = _params_path(dataset_name)
    with open(pf, 'w') as f:
        json.dump(params, f, indent=2)
    return f"✅ 参数已保存: {pf}\n{json.dumps(params, indent=2)}"

def apply_registration(dataset_name, cx, cy, scale, angle, alpha):

    if not dataset_name:
        return None, "请先选择数据集"
    if cx is None or cy is None:
        return None, "请先点击设置位置"
    md = _get_maze_channels()
    if md is None:
        return None, f"迷宫图不存在: {MAZE_PATH}"
    bgr, a, gray = md
    mh, mw = bgr.shape[:2]

    cam_bgr, _ = _first_cam(dataset_name)
    if cam_bgr is None:
        return None, "未找到图像"
    h, w = cam_bgr.shape[:2]

    M = _build_affine(mw, mh, cx, cy, scale, angle)

    wall_src = ((a > 50) & (gray < 100)).astype(np.uint8) * 255
    wall_mask = (cv2.warpAffine(wall_src, M, (w, h)) > 128).astype(np.uint8) * 255
    corr_src = ((a > 50) & (gray >= 100)).astype(np.uint8) * 255
    corr_mask = (cv2.warpAffine(corr_src, M, (w, h)) > 128).astype(np.uint8) * 255

    out_dir = os.path.join(MASK_ALIGN_DIR, dataset_name, 'registration_output')
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "maze_wall_mask.png"), wall_mask)
    cv2.imwrite(os.path.join(out_dir, "maze_corridor_mask.png"), corr_mask)

    H = np.vstack([M, [0, 0, 1]])
    np.save(os.path.join(out_dir, "homography.npy"), H)

    overlay_rgb = _overlay(cam_bgr, cx, cy, scale, angle, alpha)
    cv2.imwrite(os.path.join(out_dir, "registered.png"),
                cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    params = dict(cx=float(cx), cy=float(cy),
                  scale=float(scale), angle_deg=float(angle),
                  maze_path=MAZE_PATH)
    with open(_params_path(dataset_name), 'w') as f:
        json.dump(params, f, indent=2)

    info = (
        f"✅ 配准完成!\n"
        f"输出: {out_dir}\n"
        f"  ├── maze_wall_mask.png\n"
        f"  ├── maze_corridor_mask.png\n"
        f"  ├── homography.npy (3×3)\n"
        f"  ├── registered.png\n"
        f"  └── maze_params.json"
    )
    return overlay_rgb, info

def on_maze_click_target(dataset_name, target_selector, targets_dict, evt: gr.SelectData):

    x, y = evt.index                    
    if targets_dict is None:
        targets_dict = {}
    target_name = target_selector or 'A'
    targets_dict[target_name] = [int(y), int(x)]

    maze_rgb = _load_maze_rgb()
    info_lines = [f"Target {target_name} 设置为: ({y}, {x})  [迷宫像素坐标 (y, x)]"]
    info_lines.append("--- 已设定的目标 ---")
    for k in sorted(targets_dict.keys()):
        ty, tx = targets_dict[k]
        info_lines.append(f"  Target {k}: ({ty}, {tx})")

    info = "\n".join(info_lines)

    if maze_rgb is not None:
        annotated = maze_rgb.copy()
        _TARGET_COLORS = {'A': (255, 0, 0), 'B': (0, 180, 0), 'C': (0, 80, 255)}
        for k in sorted(targets_dict.keys()):
            ty, tx = targets_dict[k]
            c = _TARGET_COLORS.get(k, (255, 255, 0))
            cv2.drawMarker(annotated, (tx, ty), c,
                           cv2.MARKER_STAR, 30, 3)
            cv2.putText(annotated, f"Target {k} ({ty},{tx})", (tx + 15, ty - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        return targets_dict, info, annotated

    return targets_dict, info, None

_REG_CX = 624.0
_REG_CY = 578.0
_REG_SCALE = 0.565
_REG_ANGLE = 0.0

def generate_maze_features(
    dataset_name,
    targets_dict,
    active_target_name,
    bif_cluster_radius=25,
    boundary_margin=30,
    curvature_arc_length=30,
    curvature_smooth_sigma=3,
    progress=gr.Progress(),
):

    if not MAZE_EXTRACTOR_AVAILABLE:
        return "❌ MazeFeatureExtractor 依赖未安装，请安装 `iros[vision]`。"

    if not dataset_name:
        return "❌ 请先选择数据集"

    if not targets_dict:
        return "❌ 请先在迷宫图上点击设定至少一个 Target 位置"

    if not os.path.exists(MAZE_PATH):
        return f"❌ 迷宫图不存在: {MAZE_PATH}"

    progress(0.1, desc="加载迷宫图...")
    try:
        free_mask, original_img = load_and_preprocess_maze(MAZE_PATH)
    except Exception as e:
        return f"❌ 加载迷宫图失败: {e}"

    targets = {}
    for name, (ty, tx) in targets_dict.items():
        targets[name] = (int(ty), int(tx))

    start_point = (75, 75)

    n_targets = len(targets)
    progress(0.2, desc=f"构建骨架 + 计算 {n_targets} 个目标特征...")
    try:
        extractor = MazeFeatureExtractor(
            free_mask, targets, start_point,
            pixel_scale=1.0,                      
            bifurcation_cluster_radius=int(bif_cluster_radius),
            boundary_margin=int(boundary_margin),
            curvature_arc_length=int(curvature_arc_length),
            curvature_smooth_sigma=curvature_smooth_sigma,
        )
    except Exception as e:
        return f"❌ MazeFeatureExtractor 初始化失败: {e}"

    progress(0.8, desc="保存 pkl...")
    out_dir = os.path.join(MASK_ALIGN_DIR, dataset_name, 'registration_output')
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, 'maze_features.pkl')
    extractor.save(pkl_path)

    params_file = _params_path(dataset_name)
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            params = json.load(f)
    else:
        params = {}

    for name, (ty, tx) in targets.items():
        params[f'target_{name}'] = [int(ty), int(tx)]

    params['active_target'] = active_target_name or list(targets.keys())[0]

    if 'A' in targets:
        params['target_A'] = list(targets['A'])

    params['cx'] = _REG_CX
    params['cy'] = _REG_CY
    params['scale'] = _REG_SCALE
    params['angle_deg'] = _REG_ANGLE
    params['maze_path'] = MAZE_PATH

    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)

    progress(1.0, desc="完成!")

    target_info_lines = []
    for name in sorted(targets.keys()):
        ty, tx = targets[name]
        test_feat = extractor.query((ty, tx), name, normalize=False)
        marker = " ← 当前" if name == params['active_target'] else ""
        target_info_lines.append(
            f"\n[Target {name}{marker}] ({ty}, {tx})\n"
            f"  D_geo:  {test_feat['D_geo']:.2f}\n"
            f"  d_wall: {test_feat['d_wall']:.2f}\n"
            f"  kappa:  {test_feat['kappa']:.6f}\n"
            f"  d_bif:  {test_feat['d_bif']:.2f}\n"
            f"  width:  {test_feat['width']:.2f}"
        )

    targets_str = ", ".join(f"{k}=({v[0]},{v[1]})" for k, v in sorted(targets.items()))
    info = (
        f"✅ MazeFeatureExtractor 生成完成!\n"
        f"{'=' * 50}\n"
        f"迷宫尺寸: {free_mask.shape}\n"
        f"Targets ({n_targets}): {targets_str}\n"
        f"Active Target: {params['active_target']}\n"
        f"骨架节点数: {len(extractor.G.nodes)}\n"
        f"分叉点数:   {len(extractor.bifurcations)}\n"
        f"端点数:     {len(extractor.endpoints)}\n"
        f"\n保存到: {pkl_path}\n"
        f"参数写入: {params_file}\n"
        + "\n".join(target_info_lines)
    )
    return info
