#!/usr/bin/env python3

import os
import sys
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np
import gradio as gr
import pandas as pd

from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from tools.annotation import template_alignment_ui as ta
from iros.paths import DATA_ROOT

SAM2_DATASET_DIR = str(DATA_ROOT)
DEFAULT_TEMPLATE = ta.DEFAULT_TEMPLATE

_DATASET_CACHE: Dict[str, Dict[str, Any]] = {}

def _list_mask_align_datasets():

    if not os.path.isdir(SAM2_DATASET_DIR):
        return []
    return sorted(
        d
        for d in os.listdir(SAM2_DATASET_DIR)
        if os.path.isdir(os.path.join(SAM2_DATASET_DIR, d, "images_vis_template_only"))
    )

def _load_dataset_meta(dataset_name: str) -> Dict[str, Any]:

    if dataset_name in _DATASET_CACHE:
        return _DATASET_CACHE[dataset_name]

    ds_root = os.path.join(SAM2_DATASET_DIR, dataset_name)
    images_dir = os.path.join(ds_root, "images")
    vis_tpl_dir = os.path.join(ds_root, "images_vis_template_only")
    csv_path = os.path.join(ds_root, "tracking_results.csv")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images 目录不存在: {images_dir}")
    if not os.path.isdir(vis_tpl_dir):
        raise FileNotFoundError(f"images_vis_template_only 目录不存在: {vis_tpl_dir}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"tracking_results.csv 不存在: {csv_path}")

    frame_files = sorted(
        f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")
    )
    frame_files.sort(key=lambda p: int(os.path.splitext(p)[0]))

    df = pd.read_csv(csv_path)

    tpl_bgr, tpl_binary, tpl_contour, tpl_center = ta.load_template(DEFAULT_TEMPLATE)
    if tpl_binary is None or tpl_center is None:
        raise RuntimeError(f"模板加载失败: {DEFAULT_TEMPLATE}")
    tpl_endpoints = ta.get_template_endpoints(tpl_binary)

    scale_est = 0.075      
    if tpl_endpoints is not None:
        tpl_dist = float(tpl_endpoints.get("dist", 0.0)) or 0.0
        if tpl_dist > 0:
            for _, row in df.iterrows():
                bx, by = float(row["bottom_x"]), float(row["bottom_y"])
                tx, ty = float(row["top_x"]), float(row["top_y"])
                dist_frame = float(np.hypot(bx - tx, by - ty))
                if dist_frame > 1e-3:
                    scale_est = dist_frame / tpl_dist
                    break

    meta = dict(
        root=ds_root,
        images_dir=images_dir,
        vis_tpl_dir=vis_tpl_dir,
        csv_path=csv_path,
        frames=frame_files,
        df=df,
        tpl_bgr=tpl_bgr,
        tpl_binary=tpl_binary,
        tpl_contour=tpl_contour,
        tpl_center=tpl_center,
        tpl_endpoints=tpl_endpoints,
        scale_est=float(scale_est),
    )
    _DATASET_CACHE[dataset_name] = meta
    return meta

def _get_df_row(meta: Dict[str, Any], frame_idx: int) -> Optional[pd.Series]:

    df: pd.DataFrame = meta["df"]
    rows = df[df["frame"] == frame_idx]
    if rows.empty:
        return None
    return rows.iloc[0]

def _build_overlay_for_frame(
    dataset_name: str,
    frame_idx: int,
    pos_x: float,
    pos_y: float,
    scale: float,
    angle_deg: float,
    dilate_px: int,
) -> Tuple[np.ndarray, np.ndarray, str]:

    meta = _load_dataset_meta(dataset_name)
    frames = meta["frames"]
    images_dir = meta["images_dir"]
    tpl_binary = meta["tpl_binary"]
    tpl_center = meta["tpl_center"]

    if not frames:
        raise RuntimeError("该数据集下没有 images/*.jpg 帧")

    frame_idx_clamped = max(0, min(int(frame_idx), len(frames) - 1))
    img_name = frames[frame_idx_clamped]
    img_path = os.path.join(images_dir, img_name)
    frame_bgr = cv2.imread(img_path)
    if frame_bgr is None:
        raise RuntimeError(f"读取图像失败: {img_path}")

    fh, fw = frame_bgr.shape[:2]

    warped_tpl = ta.warp_template(
        tpl_binary,
        tpl_center,
        float(pos_x),
        float(pos_y),
        float(scale),
        float(angle_deg),
        fw,
        fh,
        dilate_px=int(dilate_px),
    )
    vis_rgb = ta.build_template_only_vis(frame_bgr, warped_tpl)

    info = (
        f"帧 {frame_idx_clamped}/{len(frames)-1} | "
        f"center=({pos_x:.1f}, {pos_y:.1f}), "
        f"scale={scale:.4f}, angle={angle_deg:.1f}°, "
        f"dilate={dilate_px}px"
    )
    return vis_rgb, warped_tpl, info

def _load_existing_vis(dataset_name: str, frame_idx: int) -> Optional[np.ndarray]:

    meta = _load_dataset_meta(dataset_name)
    frames = meta["frames"]
    vis_tpl_dir = meta["vis_tpl_dir"]
    if not frames:
        return None
    frame_idx_clamped = max(0, min(int(frame_idx), len(frames) - 1))
    img_name = f"{frame_idx_clamped:05d}.jpg"
    path = os.path.join(vis_tpl_dir, img_name)
    if not os.path.isfile(path):
        return None
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _update_tracking_row_from_mask(
    meta: Dict[str, Any],
    frame_idx: int,
    warped_tpl: np.ndarray,
):

    df: pd.DataFrame = meta["df"]
    rows = df[df["frame"] == frame_idx]
    if rows.empty:

        return

    idx = rows.index[0]

    if warped_tpl is None or not np.any(warped_tpl > 0):

        df.loc[idx, ["bottom_x", "bottom_y", "top_x", "top_y",
                     "centroid_x", "centroid_y", "angle_deg", "method"]] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            "manual_adjust_empty",
        ]
        return

    fh, fw = warped_tpl.shape[:2]

    tpl_ep_frame = ta.find_endpoint_circles(warped_tpl)
    tpl_pose = ta.get_mask_pose(warped_tpl)

    if tpl_ep_frame is not None:
        ep1 = np.array(tpl_ep_frame["p1"])
        ep2 = np.array(tpl_ep_frame["p2"])

        if ep1[1] >= ep2[1]:
            bot, top = ep1, ep2
        else:
            bot, top = ep2, ep1
        bottom_x, bottom_y = float(bot[0]), float(bot[1])
        top_x, top_y = float(top[0]), float(top[1])
    else:
        bottom_x = bottom_y = top_x = top_y = 0.0

    if tpl_pose is not None:
        centroid_x = float(tpl_pose["cx"])
        centroid_y = float(tpl_pose["cy"])
        angle_deg = float(tpl_pose["angle_deg"])
    else:

        ys, xs = np.where(warped_tpl > 0)
        if len(xs) > 0:
            centroid_x = float(xs.mean())
            centroid_y = float(ys.mean())
        else:
            centroid_x = fw / 2.0
            centroid_y = fh / 2.0
        angle_deg = 0.0

    df.loc[idx, ["bottom_x", "bottom_y", "top_x", "top_y",
                 "centroid_x", "centroid_y", "angle_deg", "method"]] = [
        bottom_x,
        bottom_y,
        top_x,
        top_y,
        centroid_x,
        centroid_y,
        angle_deg,
        "manual_adjust",
    ]

def on_dataset_change(dataset_name: str):

    if not dataset_name:
        return (
            None,
            None,
            "请选择数据集",
            None,
            None,
            gr.update(minimum=0, maximum=0, value=0),
            gr.update(),
            gr.update(),
        )

    try:
        meta = _load_dataset_meta(dataset_name)
    except Exception as e:
        return (
            None,
            None,
            f"加载数据集失败: {e}",
            None,
            None,
            gr.update(minimum=0, maximum=0, value=0),
            gr.update(),
            gr.update(),
        )

    frames = meta["frames"]
    n_frames = len(frames)
    if n_frames == 0:
        return (
            None,
            None,
            "该数据集下 images/ 为空",
            None,
            None,
            gr.update(minimum=0, maximum=0, value=0),
            gr.update(),
            gr.update(),
        )

    frame_idx = 0
    row = _get_df_row(meta, frame_idx)
    if row is not None:
        pos_x = float(row["centroid_x"])
        pos_y = float(row["centroid_y"])
        angle_deg = float(row["angle_deg"])
        method = str(row.get("method", ""))
    else:

        img_path = os.path.join(meta["images_dir"], frames[0])
        bgr = cv2.imread(img_path)
        if bgr is None:
            h = w = 0
        else:
            h, w = bgr.shape[:2]
        pos_x, pos_y, angle_deg, method = w / 2.0, h / 2.0, 0.0, "none"

    scale_est = meta["scale_est"]
    new_vis, _, info = _build_overlay_for_frame(
        dataset_name,
        frame_idx,
        pos_x,
        pos_y,
        scale_est,
        angle_deg,
        dilate_px=3,
    )
    old_vis = _load_existing_vis(dataset_name, frame_idx)

    info = (
        info
        + f"\n数据集: {dataset_name}, 帧数: {n_frames}"
        + f"\ntracking method: {method}"
        + f"\n估计全局缩放 scale ≈ {scale_est:.4f}"
    )

    return (
        new_vis,
        old_vis,
        info,
        float(pos_x),
        float(pos_y),
        gr.update(minimum=0, maximum=max(n_frames - 1, 0), value=0, step=1),
        gr.update(value=angle_deg),
        gr.update(value=scale_est),
    )

def on_frame_change(
    dataset_name: str,
    frame_idx: int,
    scale: float,
    dilate_px: int,
):

    if not dataset_name:
        return None, None, "请先选择数据集", None, None, gr.update()

    meta = _load_dataset_meta(dataset_name)
    frames = meta["frames"]
    n_frames = len(frames)
    if n_frames == 0:
        return None, None, "该数据集下 images/ 为空", None, None, gr.update()

    frame_idx = max(0, min(int(frame_idx), n_frames - 1))
    row = _get_df_row(meta, frame_idx)
    if row is not None:
        pos_x = float(row["centroid_x"])
        pos_y = float(row["centroid_y"])
        angle_deg = float(row["angle_deg"])
        method = str(row.get("method", ""))
    else:

        img_path = os.path.join(meta["images_dir"], frames[frame_idx])
        bgr = cv2.imread(img_path)
        if bgr is None:
            h = w = 0
        else:
            h, w = bgr.shape[:2]
        pos_x, pos_y, angle_deg, method = w / 2.0, h / 2.0, 0.0, "none"

    new_vis, _, info = _build_overlay_for_frame(
        dataset_name,
        frame_idx,
        pos_x,
        pos_y,
        scale,
        angle_deg,
        dilate_px=int(dilate_px),
    )
    old_vis = _load_existing_vis(dataset_name, frame_idx)

    info = (
        info
        + f"\ntracking_results: method={method}"
        + f"\n(可通过点击 / 滑条微调后再点 ✅ 保存)"
    )

    return (
        new_vis,
        old_vis,
        info,
        float(pos_x),
        float(pos_y),
        gr.update(value=angle_deg),
    )

def on_click_image(
    dataset_name: str,
    frame_idx: int,
    pos_x: Optional[float],
    pos_y: Optional[float],
    scale: float,
    angle_deg: float,
    dilate_px: int,
    evt: gr.SelectData,
):

    if not dataset_name:
        return None, "请先选择数据集", pos_x, pos_y
    x, y = float(evt.index[0]), float(evt.index[1])
    new_vis, _, info = _build_overlay_for_frame(
        dataset_name,
        frame_idx,
        x,
        y,
        scale,
        angle_deg,
        dilate_px=int(dilate_px),
    )
    info = info + f"\n点击设中心: ({x:.1f}, {y:.1f})"
    return new_vis, info, float(x), float(y)

def on_slider_change(
    dataset_name: str,
    frame_idx: int,
    pos_x: Optional[float],
    pos_y: Optional[float],
    scale: float,
    angle_deg: float,
    dilate_px: int,
):

    if not dataset_name:
        return None, "请先选择数据集"
    if pos_x is None or pos_y is None:
        return None, "请先点击图片设置一个中心点"

    new_vis, _, info = _build_overlay_for_frame(
        dataset_name,
        frame_idx,
        float(pos_x),
        float(pos_y),
        scale,
        angle_deg,
        dilate_px=int(dilate_px),
    )
    info = info + "\n(仅预览，尚未写回磁盘 / CSV，需点 ✅ 才会保存)"
    return new_vis, info

def on_confirm_and_next(
    dataset_name: str,
    frame_idx: int,
    pos_x: Optional[float],
    pos_y: Optional[float],
    scale: float,
    angle_deg: float,
    dilate_px: int,
):

    if not dataset_name:
        return None, None, "请先选择数据集", gr.update()
    if pos_x is None or pos_y is None:
        return None, None, "请先点击图片设置中心点，再保存", gr.update()

    meta = _load_dataset_meta(dataset_name)
    frames = meta["frames"]
    n_frames = len(frames)
    frame_idx = max(0, min(int(frame_idx), n_frames - 1))

    new_vis_curr, warped_tpl_curr, info_curr = _build_overlay_for_frame(
        dataset_name,
        frame_idx,
        float(pos_x),
        float(pos_y),
        scale,
        angle_deg,
        dilate_px=int(dilate_px),
    )

    img_name = f"{frame_idx:05d}.jpg"
    out_path = os.path.join(meta["vis_tpl_dir"], img_name)
    cv2.imwrite(out_path, cv2.cvtColor(new_vis_curr, cv2.COLOR_RGB2BGR))

    _update_tracking_row_from_mask(meta, frame_idx, warped_tpl_curr)

    meta["df"].to_csv(meta["csv_path"], index=False)

    if frame_idx < n_frames - 1:
        next_idx = frame_idx + 1
        row_next = _get_df_row(meta, next_idx)
        if row_next is not None:
            next_pos_x = float(row_next["centroid_x"])
            next_pos_y = float(row_next["centroid_y"])
            next_angle = float(row_next["angle_deg"])
        else:

            next_pos_x, next_pos_y, next_angle = float(pos_x), float(
                pos_y
            ), float(angle_deg)

        new_vis_next, _, info_next = _build_overlay_for_frame(
            dataset_name,
            next_idx,
            next_pos_x,
            next_pos_y,
            scale,
            next_angle,
            dilate_px=int(dilate_px),
        )
        old_vis_next = _load_existing_vis(dataset_name, next_idx)

        info = (
            f"✅ 已保存帧 {frame_idx}: {out_path}\n"
            f"并更新 tracking_results.csv 第 {frame_idx} 行 (method=manual_adjust)\n\n"
            f"现在显示下一帧 {next_idx}:\n"
            f"{info_next}"
        )

        return (
            new_vis_next,
            old_vis_next,
            info,
            gr.update(value=next_idx),
        )
    else:

        old_vis_last = _load_existing_vis(dataset_name, frame_idx)
        info = (
            f"✅ 已保存最后一帧 {frame_idx}: {out_path}\n"
            f"并更新 tracking_results.csv 第 {frame_idx} 行 (method=manual_adjust)\n"
            f"已经是最后一帧，可以手动拖动滑条回看前面的帧。"
        )
        return (
            new_vis_curr,
            old_vis_last,
            info,
            gr.update(value=frame_idx),
        )

def build_app():
    ds_choices = _list_mask_align_datasets()

    with gr.Blocks(title="mask_align 结果复查 & 手动微调") as app:
        gr.Markdown("## Step 3.2 结果复查：逐帧检查 mask 模板叠加效果")
        gr.Markdown(
            "- **左图：当前参数下重新渲染的模板叠加 (可点击/拖动/调角度)**\n"
            "- **右图：磁盘上已有的 `images_vis_template_only` 原始可视化**\n"
            "- 若不满意：像 3.1 一样点击设置中心 + 调节旋转/缩放/膨胀 → 点“✅ 当前帧通过 & 保存(下一张)”\n"
            "- 对满意的帧，直接点“✅”即可跳到下一张\n"
        )

        with gr.Row():
            dataset_dd = gr.Dropdown(
                choices=ds_choices,
                label="选择 mask_align 数据集",
                info="来自 IROS_DATA_ROOT",
            )
            frame_slider = gr.Slider(
                minimum=0,
                maximum=0,
                value=0,
                step=1,
                label="帧索引",
                info="从 0 到 N-1，建议从头开始依次检查",
            )

        pos_x_state = gr.State(None)
        pos_y_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=2):
                new_img = gr.Image(
                    label="重新渲染的模板叠加 (点击设置中心)",
                    interactive=True,
                )
            with gr.Column(scale=2):
                old_img = gr.Image(
                    label="当前磁盘上的 images_vis_template_only 结果 (只读)",
                    interactive=False,
                )

        info_box = gr.Textbox(label="信息 / 状态", lines=6, interactive=False)

        with gr.Row():
            scale_slider = gr.Slider(
                0.01, 0.5, 0.075, step=0.005, label="缩放 scale", info="全局缩放系数"
            )
            angle_slider = gr.Slider(
                -180, 180, 0.0, step=0.5, label="旋转 (度)", info="围绕模板质心旋转"
            )
            dilate_slider = gr.Slider(
                0, 10, 3, step=1, label="模板膨胀 (px)", info="加粗中间细杆"
            )

        with gr.Row():
            confirm_btn = gr.Button("✅ 当前帧通过 & 保存(下一张)", variant="primary")

        dataset_dd.change(
            fn=on_dataset_change,
            inputs=[dataset_dd],
            outputs=[
                new_img,
                old_img,
                info_box,
                pos_x_state,
                pos_y_state,
                frame_slider,
                angle_slider,
                scale_slider,
            ],
        )

        frame_slider.release(
            fn=on_frame_change,
            inputs=[dataset_dd, frame_slider, scale_slider, dilate_slider],
            outputs=[new_img, old_img, info_box, pos_x_state, pos_y_state, angle_slider],
        )

        new_img.select(
            fn=on_click_image,
            inputs=[
                dataset_dd,
                frame_slider,
                pos_x_state,
                pos_y_state,
                scale_slider,
                angle_slider,
                dilate_slider,
            ],
            outputs=[new_img, info_box, pos_x_state, pos_y_state],
        )

        def _bind_slider(ctrl):
            ctrl.release(
                fn=on_slider_change,
                inputs=[
                    dataset_dd,
                    frame_slider,
                    pos_x_state,
                    pos_y_state,
                    scale_slider,
                    angle_slider,
                    dilate_slider,
                ],
                outputs=[new_img, info_box],
            )

        for _ctrl in [scale_slider, angle_slider, dilate_slider]:
            _bind_slider(_ctrl)

        confirm_btn.click(
            fn=on_confirm_and_next,
            inputs=[
                dataset_dd,
                frame_slider,
                pos_x_state,
                pos_y_state,
                scale_slider,
                angle_slider,
                dilate_slider,
            ],
            outputs=[new_img, old_img, info_box, frame_slider],
        )

    return app

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="mask_align 结果复查 & 手动微调 UI")
    parser.add_argument(
        "--port", type=int, default=7864, help="端口号 (默认: 7864)"
    )
    args = parser.parse_args()

    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
