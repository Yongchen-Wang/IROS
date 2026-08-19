import cv2
import numpy as np
import os
import csv
import glob
from iros.paths import ASSET_ROOT, SHARED_CONTROL_DATA_ROOT

DIR = str(SHARED_CONTROL_DATA_ROOT)
LABEL_DIR = os.path.join(DIR, "labels")
IMG_DIR = os.path.join(DIR, "images_vis")
OUT_DIR = os.path.join(DIR, "registration_output")
MAZE_PATH = str(ASSET_ROOT / "maze.png")
H_PATH = os.path.join(OUT_DIR, "homography.npy")

IMG_W, IMG_H = 410, 330
STEP = 4
RAY_MAX = 200

def build_masks():

    maze = cv2.imread(MAZE_PATH, cv2.IMREAD_UNCHANGED)
    a = maze[:, :, 3]
    gray = cv2.cvtColor(maze[:, :, :3], cv2.COLOR_BGR2GRAY)
    H = np.load(H_PATH)

    wall_src = ((a > 50) & (gray < 100)).astype(np.uint8) * 255
    wall_mask = cv2.warpPerspective(wall_src, H, (IMG_W, IMG_H))
    wall_mask = (wall_mask > 128).astype(np.uint8) * 255

    corr_src = ((a > 50) & (gray >= 100)).astype(np.uint8) * 255
    corr_mask = cv2.warpPerspective(corr_src, H, (IMG_W, IMG_H))
    corr_mask = (corr_mask > 128).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(OUT_DIR, "maze_wall_mask.png"), wall_mask)
    cv2.imwrite(os.path.join(OUT_DIR, "maze_corridor_mask.png"), corr_mask)
    return wall_mask, corr_mask

def parse_yolo_seg(label_path):

    polygons = []
    if not os.path.exists(label_path):
        return polygons
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = list(map(float, parts[1:]))
            pts = [[coords[i] * IMG_W, coords[i + 1] * IMG_H]
                   for i in range(0, len(coords), 2)]
            polygons.append(np.array(pts, dtype=np.float32))
    return polygons

def raycast(cx, cy, dx, dy, mask, hit_value, max_dist=RAY_MAX):

    h, w = mask.shape
    for s in range(1, max_dist):
        x = int(round(cx + dx * s))
        y = int(round(cy + dy * s))
        if x < 0 or x >= w or y < 0 or y >= h:
            return float(s)
        if mask[y, x] == hit_value:
            return float(s)
    return float(max_dist)

def heading_dirs(vx, vy):

    sp = np.sqrt(vx**2 + vy**2)
    if sp < 0.01:
        vx, vy, sp = 0.0, 1.0, 1.0
    hx, hy = vx / sp, vy / sp
    lx, ly = -hy, hx                  
    rx, ry = hy, -hx                   
    return (hx, hy), (lx, ly), (rx, ry)

def angle_diff(a1, a2):

    d = a1 - a2
    return (d + np.pi) % (2 * np.pi) - np.pi

def main():
    if not os.path.exists(H_PATH):
        print("请先运行 maze_registration.py 完成配准")
        return

    wall_mask, corr_mask = build_masks()
    print(f"墙壁像素: {np.sum(wall_mask > 0)}, 通道像素: {np.sum(corr_mask > 0)}")

    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.txt")))
    print(f"共 {len(label_files)} 帧标签")

    all_pos = {}
    for lf in label_files:
        fid = int(os.path.splitext(os.path.basename(lf))[0])
        polygons = parse_yolo_seg(lf)
        if polygons:
            all_pos[fid] = tuple(polygons[0].mean(axis=0))

    sampled = sorted(fid for fid in all_pos if fid % STEP == 0)

    results = []
    prev_heading = None

    for i, fid in enumerate(sampled):
        cx, cy = all_pos[fid]

        vx, vy, speed = 0.0, 0.0, 0.0
        if i > 0:
            px, py = all_pos[sampled[i - 1]]
            vx = (cx - px) / STEP
            vy = (cy - py) / STEP
            speed = np.sqrt(vx**2 + vy**2)

        heading = np.arctan2(vy, vx) if speed > 0.01 else (prev_heading if prev_heading is not None else np.pi / 2)
        heading_deg = np.degrees(heading)

        curvature = 0.0
        if prev_heading is not None and speed > 0.01:
            dtheta = angle_diff(heading, prev_heading)
            ds = speed * STEP      
            curvature = dtheta / ds if ds > 0.1 else 0.0
        prev_heading = heading

        _, (lx, ly), (rx, ry) = heading_dirs(vx, vy)

        dist_l_wall = raycast(cx, cy, lx, ly, wall_mask, 255)
        dist_r_wall = raycast(cx, cy, rx, ry, wall_mask, 255)

        dist_l_corr = raycast(cx, cy, lx, ly, corr_mask, 0)
        dist_r_corr = raycast(cx, cy, rx, ry, corr_mask, 0)
        corridor_width = dist_l_corr + dist_r_corr

        results.append({
            "frame": fid,
            "cx": round(float(cx), 2),
            "cy": round(float(cy), 2),
            "vx": round(float(vx), 3),
            "vy": round(float(vy), 3),
            "speed": round(float(speed), 3),
            "heading_deg": round(float(heading_deg), 1),
            "curvature": round(float(curvature), 4),
            "dist_l_wall": round(float(dist_l_wall), 1),
            "dist_r_wall": round(float(dist_r_wall), 1),
            "dist_l_corr": round(float(dist_l_corr), 1),
            "dist_r_corr": round(float(dist_r_corr), 1),
            "corridor_w": round(float(corridor_width), 1),
        })

    csv_path = os.path.join(OUT_DIR, "robot_metrics.csv")
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    hdr = (f"{'frm':>4} {'cx':>6} {'cy':>6} {'spd':>5} {'hdg':>6} "
           f"{'curv':>7} {'Lwall':>5} {'Rwall':>5} {'Lcor':>5} {'Rcor':>5} {'cW':>4}")
    print(f"\n{hdr}\n{'-'*len(hdr)}")
    for r in results:
        print(f"{r['frame']:>4} {r['cx']:>6} {r['cy']:>6} {r['speed']:>5} "
              f"{r['heading_deg']:>6} {r['curvature']:>7} "
              f"{r['dist_l_wall']:>5} {r['dist_r_wall']:>5} "
              f"{r['dist_l_corr']:>5} {r['dist_r_corr']:>5} "
              f"{r['corridor_w']:>4}")
    print(f"\n结果已保存: {csv_path}")

    cam = cv2.imread(os.path.join(IMG_DIR, "00000.jpg"))
    vis = cam.copy()

    c1, _ = cv2.findContours(corr_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, c1, -1, (200, 180, 0), 1)
    c2, _ = cv2.findContours(wall_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, c2, -1, (0, 0, 200), 1)

    curvatures = [abs(r["curvature"]) for r in results]
    max_k = max(curvatures[1:]) if len(curvatures) > 1 else 1.0
    max_k = max(max_k, 0.001)

    for j in range(1, len(results)):
        p1 = (int(results[j-1]["cx"]), int(results[j-1]["cy"]))
        p2 = (int(results[j]["cx"]), int(results[j]["cy"]))

        k_norm = min(abs(results[j]["curvature"]) / max_k, 1.0)
        color = (0, int(255 * (1 - k_norm)), int(255 * k_norm))
        cv2.line(vis, p1, p2, color, 2)

    for j in range(0, len(results), 4):
        r = results[j]
        pt = (int(r["cx"]), int(r["cy"]))
        sp = r["speed"]
        vx, vy = r["vx"], r["vy"]
        (hx, hy), (lx, ly), (rx, ry) = heading_dirs(vx, vy)

        arr_len = 15
        pt_h = (int(r["cx"] + hx * arr_len), int(r["cy"] + hy * arr_len))
        cv2.arrowedLine(vis, pt, pt_h, (255, 255, 255), 1, tipLength=0.3)

        dl, dr = r["dist_l_corr"], r["dist_r_corr"]
        pt_l = (int(r["cx"] + lx * dl), int(r["cy"] + ly * dl))
        pt_r = (int(r["cx"] + rx * dr), int(r["cy"] + ry * dr))
        cv2.line(vis, pt_l, pt_r, (255, 200, 50), 1)
        cv2.circle(vis, pt, 3, (0, 255, 255), -1)

    vis_path = os.path.join(OUT_DIR, "trajectory.png")
    cv2.imwrite(vis_path, vis)
    print(f"轨迹可视化: {vis_path}")

if __name__ == "__main__":
    main()
