import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

from ..paths import OUTPUT_ROOT, SHARED_CONTROL_DATA_ROOT

def load_analysis_data(json_path: str) -> Dict:

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_robot_metrics(csv_path: str) -> Optional[pd.DataFrame]:

    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        return None

def segment_by_curvature(
    detections: List[Dict],
    robot_metrics: Optional[pd.DataFrame] = None,
    curvature_threshold: float = 0.1
) -> List[Tuple[int, int, str]]:

    if robot_metrics is None:

        return [(0, len(detections) - 1, 'unknown')]

    segments = []
    current_start = 0
    current_type = 'straight'

    for i in range(len(detections)):
        frame_id = detections[i]['frame_id']

        metric_row = robot_metrics[robot_metrics['frame'] == frame_id]

        if len(metric_row) == 0:
            continue

        curvature = abs(metric_row.iloc[0]['curvature'])

        if curvature < curvature_threshold:
            seg_type = 'straight'
        else:

            curvature_val = metric_row.iloc[0]['curvature']
            seg_type = 'left_turn' if curvature_val > 0 else 'right_turn'

        if seg_type != current_type and i > current_start:
            segments.append((current_start, i - 1, current_type))
            current_start = i
            current_type = seg_type

    if current_start < len(detections):
        segments.append((current_start, len(detections) - 1, current_type))

    return segments

def extract_arm_trajectories(detections: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    left_centroids = []
    right_centroids = []
    frame_ids = []

    for det in detections:
        if det['left_rod'] is not None:
            left_centroids.append(det['left_rod']['centroid'])
        else:
            left_centroids.append([np.nan, np.nan])

        if det['right_rod'] is not None:
            right_centroids.append(det['right_rod']['centroid'])
        else:
            right_centroids.append([np.nan, np.nan])

        frame_ids.append(det['frame_id'])

    return (
        np.array(left_centroids),
        np.array(right_centroids),
        np.array(frame_ids)
    )

def compute_velocity_and_acceleration(
    positions: np.ndarray,
    frame_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)

    for i in range(1, len(positions)):
        if not np.isnan(positions[i]).any() and not np.isnan(positions[i-1]).any():
            dt = frame_ids[i] - frame_ids[i-1]
            if dt > 0:
                velocities[i] = (positions[i] - positions[i-1]) / dt

                if i > 1 and not np.isnan(velocities[i-1]).any():
                    accelerations[i] = (velocities[i] - velocities[i-1]) / dt

    return velocities, accelerations

def analyze_segment(
    detections: List[Dict],
    start_idx: int,
    end_idx: int,
    segment_type: str
) -> Dict:

    segment_detections = detections[start_idx:end_idx+1]

    left_centroids, right_centroids, frame_ids = extract_arm_trajectories(segment_detections)

    left_areas = []
    left_widths = []
    left_lengths = []
    right_areas = []
    right_widths = []
    right_lengths = []

    for det in segment_detections:
        if det['left_rod']:
            left_areas.append(det['left_rod']['area'])
            left_widths.append(det['left_rod']['width'])
            left_lengths.append(det['left_rod']['length'])
        else:
            left_areas.append(np.nan)
            left_widths.append(np.nan)
            left_lengths.append(np.nan)

        if det['right_rod']:
            right_areas.append(det['right_rod']['area'])
            right_widths.append(det['right_rod']['width'])
            right_lengths.append(det['right_rod']['length'])
        else:
            right_areas.append(np.nan)
            right_widths.append(np.nan)
            right_lengths.append(np.nan)

    left_velocities, left_accelerations = compute_velocity_and_acceleration(
        left_centroids, frame_ids
    )
    right_velocities, right_accelerations = compute_velocity_and_acceleration(
        right_centroids, frame_ids
    )

    left_speeds = np.linalg.norm(left_velocities, axis=1)
    right_speeds = np.linalg.norm(right_velocities, axis=1)

    left_accel_mags = np.linalg.norm(left_accelerations, axis=1)
    right_accel_mags = np.linalg.norm(right_accelerations, axis=1)

    return {
        'segment_type': segment_type,
        'start_frame': detections[start_idx]['frame_id'],
        'end_frame': detections[end_idx]['frame_id'],
        'n_frames': end_idx - start_idx + 1,
        'left_arm': {
            'centroids': left_centroids,
            'areas': np.array(left_areas),
            'widths': np.array(left_widths),
            'lengths': np.array(left_lengths),
            'velocities': left_velocities,
            'speeds': left_speeds,
            'accelerations': left_accelerations,
            'accel_magnitudes': left_accel_mags,
        },
        'right_arm': {
            'centroids': right_centroids,
            'areas': np.array(right_areas),
            'widths': np.array(right_widths),
            'lengths': np.array(right_lengths),
            'velocities': right_velocities,
            'speeds': right_speeds,
            'accelerations': right_accelerations,
            'accel_magnitudes': right_accel_mags,
        },
        'frame_ids': frame_ids,
    }

def visualize_segment_analysis(
    segment_analysis: Dict,
    output_path: str
):

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(
        f"双臂运动分析 - {segment_analysis['segment_type']} "
        f"(帧 {segment_analysis['start_frame']}-{segment_analysis['end_frame']})",
        fontsize=16
    )

    frame_ids = segment_analysis['frame_ids']
    left = segment_analysis['left_arm']
    right = segment_analysis['right_arm']

    ax = axes[0, 0]
    ax.plot(frame_ids, left['centroids'][:, 0], 'b-', label='左臂 X', alpha=0.7)
    ax.plot(frame_ids, right['centroids'][:, 0], 'r-', label='右臂 X', alpha=0.7)
    ax.set_xlabel('帧ID')
    ax.set_ylabel('X坐标 (像素)')
    ax.set_title('X坐标变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(frame_ids, left['centroids'][:, 1], 'b-', label='左臂 Y', alpha=0.7)
    ax.plot(frame_ids, right['centroids'][:, 1], 'r-', label='右臂 Y', alpha=0.7)
    ax.set_xlabel('帧ID')
    ax.set_ylabel('Y坐标 (像素)')
    ax.set_title('Y坐标变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(frame_ids, left['areas'], 'b-', label='左臂面积', alpha=0.7)
    ax.plot(frame_ids, right['areas'], 'r-', label='右臂面积', alpha=0.7)
    ax.set_xlabel('帧ID')
    ax.set_ylabel('面积 (像素²)')
    ax.set_title('掩码面积变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(frame_ids, left['widths'], 'b-', label='左臂宽度', alpha=0.7)
    ax.plot(frame_ids, right['widths'], 'r-', label='右臂宽度', alpha=0.7)
    ax.set_xlabel('帧ID')
    ax.set_ylabel('宽度 (像素)')
    ax.set_title('掩码宽度变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(frame_ids, left['lengths'], 'b-', label='左臂长度', alpha=0.7)
    ax.plot(frame_ids, right['lengths'], 'r-', label='右臂长度', alpha=0.7)
    ax.set_xlabel('帧ID')
    ax.set_ylabel('长度 (像素)')
    ax.set_title('掩码长度变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(frame_ids, left['speeds'], 'b-', label='左臂速度', alpha=0.7)
    ax.plot(frame_ids, right['speeds'], 'r-', label='右臂速度', alpha=0.7)
    ax.set_xlabel('帧ID')
    ax.set_ylabel('速度 (像素/帧)')
    ax.set_title('运动速度')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(frame_ids, left['velocities'][:, 0], 'b-', label='左臂 Vx', alpha=0.7)
    ax.plot(frame_ids, right['velocities'][:, 0], 'r-', label='右臂 Vx', alpha=0.7)
    ax.set_xlabel('帧ID')
    ax.set_ylabel('Vx (像素/帧)')
    ax.set_title('X方向速度')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(frame_ids, left['velocities'][:, 1], 'b-', label='左臂 Vy', alpha=0.7)
    ax.plot(frame_ids, right['velocities'][:, 1], 'r-', label='右臂 Vy', alpha=0.7)
    ax.set_xlabel('帧ID')
    ax.set_ylabel('Vy (像素/帧)')
    ax.set_title('Y方向速度')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    ax.plot(frame_ids, left['accel_magnitudes'], 'b-', label='左臂加速度', alpha=0.7)
    ax.plot(frame_ids, right['accel_magnitudes'], 'r-', label='右臂加速度', alpha=0.7)
    ax.set_xlabel('帧ID')
    ax.set_ylabel('加速度 (像素/帧²)')
    ax.set_title('运动加速度')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_summary_report(
    all_segments: List[Dict],
    output_path: str
):

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("双臂黑色掩码运动变化分析报告")
    report_lines.append("=" * 80)
    report_lines.append("")

    for i, seg in enumerate(all_segments):
        report_lines.append(f"\n段 {i+1}: {seg['segment_type']}")
        report_lines.append(f"  帧范围: {seg['start_frame']} - {seg['end_frame']} ({seg['n_frames']} 帧)")

        left = seg['left_arm']
        report_lines.append("\n  左臂统计:")
        left_x_diff = np.diff(left['centroids'][:, 0])
        left_y_diff = np.diff(left['centroids'][:, 1])
        report_lines.append(f"    位置变化: X={np.nanmean(left_x_diff):.2f} px/帧, "
                          f"Y={np.nanmean(left_y_diff):.2f} px/帧")
        report_lines.append(f"    平均速度: {np.nanmean(left['speeds']):.2f} px/帧")
        report_lines.append(f"    平均面积: {np.nanmean(left['areas']):.1f} px²")
        report_lines.append(f"    平均宽度: {np.nanmean(left['widths']):.1f} px")
        report_lines.append(f"    平均长度: {np.nanmean(left['lengths']):.1f} px")

        right = seg['right_arm']
        report_lines.append("\n  右臂统计:")
        right_x_diff = np.diff(right['centroids'][:, 0])
        right_y_diff = np.diff(right['centroids'][:, 1])
        report_lines.append(f"    位置变化: X={np.nanmean(right_x_diff):.2f} px/帧, "
                          f"Y={np.nanmean(right_y_diff):.2f} px/帧")
        report_lines.append(f"    平均速度: {np.nanmean(right['speeds']):.2f} px/帧")
        report_lines.append(f"    平均面积: {np.nanmean(right['areas']):.1f} px²")
        report_lines.append(f"    平均宽度: {np.nanmean(right['widths']):.1f} px")
        report_lines.append(f"    平均长度: {np.nanmean(right['lengths']):.1f} px")

        report_lines.append("\n  双臂对比:")
        left_speed_mean = np.nanmean(left['speeds'])
        right_speed_mean = np.nanmean(right['speeds'])
        if left_speed_mean > 0:
            report_lines.append(f"    速度比 (右/左): {right_speed_mean / left_speed_mean:.2f}")
        left_area_mean = np.nanmean(left['areas'])
        right_area_mean = np.nanmean(right['areas'])
        if left_area_mean > 0:
            report_lines.append(f"    面积比 (右/左): {right_area_mean / left_area_mean:.2f}")

    report_lines.append("\n" + "=" * 80)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print('\n'.join(report_lines))

def main():

    json_path = OUTPUT_ROOT / "navigation" / "ump_mask_analysis_robot_data_20260220_195724.json"
    robot_metrics_path = SHARED_CONTROL_DATA_ROOT / "registration_output" / "robot_metrics.csv"
    output_dir = OUTPUT_ROOT / "navigation" / "arm_motion_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载分析数据...")
    data = load_analysis_data(json_path)
    detections = data['detections']
    print(f"加载了 {len(detections)} 条检测数据")

    robot_metrics = load_robot_metrics(robot_metrics_path)
    if robot_metrics is not None:
        print(f"加载了 {len(robot_metrics)} 条运动指标数据")
    else:
        print("未找到运动指标数据，将不分段分析")

    print("分段轨迹...")
    segments = segment_by_curvature(detections, robot_metrics, curvature_threshold=0.1)
    print(f"识别到 {len(segments)} 个轨迹段")

    all_segments = []
    for i, (start_idx, end_idx, seg_type) in enumerate(segments):
        print(f"分析段 {i+1}/{len(segments)}: {seg_type} (帧 {detections[start_idx]['frame_id']}-{detections[end_idx]['frame_id']})")
        seg_analysis = analyze_segment(detections, start_idx, end_idx, seg_type)
        all_segments.append(seg_analysis)

        vis_path = output_dir / f"segment_{i+1}_{seg_type}_frames_{seg_analysis['start_frame']}_{seg_analysis['end_frame']}.png"
        visualize_segment_analysis(seg_analysis, str(vis_path))
        print(f"  已保存可视化: {vis_path}")

    report_path = output_dir / "motion_analysis_report.txt"
    generate_summary_report(all_segments, str(report_path))

    print(f"\n分析完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
