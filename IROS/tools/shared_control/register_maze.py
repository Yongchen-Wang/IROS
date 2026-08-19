#!/usr/bin/env python3

import cv2
import numpy as np
import gradio as gr
import os
import glob
from iros.paths import ASSET_ROOT, SHARED_CONTROL_DATA_ROOT

DIR = str(SHARED_CONTROL_DATA_ROOT)
MAZE_PATH = str(ASSET_ROOT / "maze.png")
CAM_PATH = os.path.join(DIR, "images_vis", "00000.jpg")
IMG_DIR = os.path.join(DIR, "images_vis")
OUT_DIR = os.path.join(DIR, "registration_output")
BATCH_DIR = os.path.join(OUT_DIR, "batch")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 165, 0)]                
LABELS = ["1", "2", "3", "4"]

def load_images():

    maze = cv2.imread(MAZE_PATH, cv2.IMREAD_UNCHANGED)
    if maze.shape[2] == 4:

        alpha = maze[:, :, 3:4].astype(np.float32) / 255
        bgr = maze[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255)
        maze_rgb = (bgr * alpha + white * (1 - alpha)).astype(np.uint8)
        maze_rgb = cv2.cvtColor(maze_rgb, cv2.COLOR_BGR2RGB)
    else:
        maze_rgb = cv2.cvtColor(maze, cv2.COLOR_BGR2RGB)

    cam = cv2.imread(CAM_PATH)
    cam_rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    return maze_rgb, cam_rgb

MAZE_ORIG, CAM_ORIG = load_images()

def draw_points(img_orig, pts):

    vis = img_orig.copy()
    h, w = vis.shape[:2]
    r = max(5, min(h, w) // 60)
    for i, (x, y) in enumerate(pts):
        color = COLORS[i % len(COLORS)]
        cv2.circle(vis, (int(x), int(y)), r, color, -1)
        cv2.circle(vis, (int(x), int(y)), r, (255, 255, 255), 2)
        cv2.putText(vis, LABELS[i], (int(x) + r + 2, int(y) + r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis

def click_maze(maze_pts, evt: gr.SelectData):

    x, y = evt.index
    if len(maze_pts) >= 4:
        maze_pts = []            
    maze_pts.append([x, y])
    vis = draw_points(MAZE_ORIG, maze_pts)
    info = f"迷宫点 ({len(maze_pts)}/4): " + ", ".join(f"({p[0]},{p[1]})" for p in maze_pts)
    return vis, maze_pts, info

def click_cam(cam_pts, evt: gr.SelectData):

    x, y = evt.index
    if len(cam_pts) >= 4:
        cam_pts = []
    cam_pts.append([x, y])
    vis = draw_points(CAM_ORIG, cam_pts)
    info = f"相机点 ({len(cam_pts)}/4): " + ", ".join(f"({p[0]},{p[1]})" for p in cam_pts)
    return vis, cam_pts, info

def do_register(maze_pts, cam_pts, alpha_val):

    if len(maze_pts) != 4 or len(cam_pts) != 4:
        return None, "请在两张图上各点击 4 个对应点"

    src = np.float32(maze_pts)
    dst = np.float32(cam_pts)
    H, _ = cv2.findHomography(src, dst)

    maze_raw = cv2.imread(MAZE_PATH, cv2.IMREAD_UNCHANGED)
    cam_bgr = cv2.imread(CAM_PATH)
    h, w = cam_bgr.shape[:2]

    if maze_raw.shape[2] == 4:
        bgr, a = maze_raw[:, :, :3], maze_raw[:, :, 3]
    else:
        bgr, a = maze_raw, np.full(maze_raw.shape[:2], 255, np.uint8)

    w_bgr = cv2.warpPerspective(bgr, H, (w, h))
    w_alpha = cv2.warpPerspective(a, H, (w, h))
    mask = (w_alpha.astype(np.float32) / 255.0)[..., None] * (alpha_val / 100.0)

    green = np.zeros_like(w_bgr)
    green[:] = (0, 128, 128)
    result = np.clip(cam_bgr * (1 - mask) + green * mask, 0, 255).astype(np.uint8)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    gray_maze = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    w_gray = cv2.warpPerspective(gray_maze, H, (w, h))
    wall_mask_src = ((a > 50) & (gray_maze < 100)).astype(np.uint8) * 255
    wall_mask = cv2.warpPerspective(wall_mask_src, H, (w, h))
    wall_mask = (wall_mask > 128).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(OUT_DIR, "maze_wall_mask.png"), wall_mask)

    cv2.imwrite(os.path.join(OUT_DIR, "registered.png"), result)
    np.save(os.path.join(OUT_DIR, "homography.npy"), H)

    info = (f"配准完成!\n"
            f"迷宫点: {src.tolist()}\n"
            f"相机点: {dst.tolist()}\n"
            f"H =\n{H}\n"
            f"迷宫墙壁mask已保存: maze_wall_mask.png")
    return result_rgb, info

def do_batch(maze_pts, cam_pts, alpha_val, progress=gr.Progress()):

    if len(maze_pts) != 4 or len(cam_pts) != 4:
        return "请先完成单帧配准（两张图各点 4 个对应点）"

    src = np.float32(maze_pts)
    dst = np.float32(cam_pts)
    H, _ = cv2.findHomography(src, dst)

    maze_raw = cv2.imread(MAZE_PATH, cv2.IMREAD_UNCHANGED)
    if maze_raw.shape[2] == 4:
        bgr, a = maze_raw[:, :, :3], maze_raw[:, :, 3]
    else:
        bgr, a = maze_raw, np.full(maze_raw.shape[:2], 255, np.uint8)

    frames = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not frames:
        return "未找到图像帧"

    os.makedirs(BATCH_DIR, exist_ok=True)

    sample = cv2.imread(frames[0])
    fh, fw = sample.shape[:2]
    w_bgr = cv2.warpPerspective(bgr, H, (fw, fh))
    w_alpha = cv2.warpPerspective(a, H, (fw, fh))
    mask = (w_alpha.astype(np.float32) / 255.0)[..., None] * (alpha_val / 100.0)
    overlay_color = np.zeros_like(w_bgr)
    overlay_color[:] = (0, 0, 255)            

    for i, fpath in enumerate(progress.tqdm(frames, desc="批量配准中")):
        cam_bgr = cv2.imread(fpath)
        result = np.clip(
            cam_bgr * (1 - mask) + overlay_color * mask, 0, 255
        ).astype(np.uint8)
        cv2.imwrite(os.path.join(BATCH_DIR, os.path.basename(fpath)), result)

    np.save(os.path.join(OUT_DIR, "homography.npy"), H)
    return f"批量配准完成! 共 {len(frames)} 帧\n输出目录: {BATCH_DIR}"

def reset_all():

    return MAZE_ORIG, CAM_ORIG, [], [], None, "已重置，请重新选点"

def build_ui():
    with gr.Blocks(title="迷宫配准") as app:
        gr.Markdown("# 迷宫配准工具\n在迷宫图和相机图上各 **点击 4 个对应特征点**，然后点击配准")

        maze_pts = gr.State([])
        cam_pts = gr.State([])

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 迷宫图 (点击选 4 个点)")
                maze_img = gr.Image(value=MAZE_ORIG, label="迷宫", interactive=False)
                maze_info = gr.Textbox(label="迷宫特征点", value="迷宫点 (0/4)")
            with gr.Column():
                gr.Markdown("### 相机图 (点击对应 4 个点)")
                cam_img = gr.Image(value=CAM_ORIG, label="相机", interactive=False)
                cam_info = gr.Textbox(label="相机特征点", value="相机点 (0/4)")

        with gr.Row():
            alpha_slider = gr.Slider(10, 100, value=70, step=5, label="叠加透明度 %")
            reg_btn = gr.Button("配准(单帧)", variant="primary", scale=2)
            batch_btn = gr.Button("传播到所有帧", variant="secondary", scale=2)
            reset_btn = gr.Button("重置", scale=1)

        gr.Markdown("### 配准结果")
        result_img = gr.Image(label="配准结果")
        result_info = gr.Textbox(label="信息", lines=5)

        maze_img.select(click_maze, [maze_pts], [maze_img, maze_pts, maze_info])
        cam_img.select(click_cam, [cam_pts], [cam_img, cam_pts, cam_info])
        reg_btn.click(do_register, [maze_pts, cam_pts, alpha_slider], [result_img, result_info])
        batch_btn.click(do_batch, [maze_pts, cam_pts, alpha_slider], [result_info])
        reset_btn.click(reset_all, [], [maze_img, cam_img, maze_pts, cam_pts, result_img, result_info])

    return app

if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7861)
