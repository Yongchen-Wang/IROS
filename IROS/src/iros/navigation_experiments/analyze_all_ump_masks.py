import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..paths import DATA_ROOT, OUTPUT_ROOT

def analyze_all_images(
    dataset_name: str,
    black_threshold: int = 30,
    min_area: int = 100,
    left_margin: int = 50,
    right_margin: int = 50,
    output_dir: str = str(OUTPUT_ROOT / "navigation"),
) -> Dict[str, Any]:

    dataset_dir = DATA_ROOT / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    img_dir = dataset_dir / "images_vis"
    if not img_dir.exists():
        img_dir = dataset_dir / "images"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found in {dataset_dir}")

    img_files = sorted([
        f for f in img_dir.iterdir() 
        if f.suffix.lower() in ('.jpg', '.png', '.jpeg')
    ])

    if not img_files:
        raise ValueError(f"No images found in {img_dir}")

    print(f"找到 {len(img_files)} 张图像")
    print(f"开始分析黑色掩码 (threshold={black_threshold}, min_area={min_area})...")

    all_detections = []
    left_rods = []
    right_rods = []

    for img_path in tqdm(img_files, desc="处理图像"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        detection = {
            'image_path': str(img_path),
            'image_name': img_path.name,
            'frame_id': int(img_path.stem) if img_path.stem.isdigit() else None,
            'left_rod': None,
            'right_rod': None,
            'total_black_pixels': int(np.sum(binary == 255)),
            'total_components': num_labels - 1,        
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
                rod_info = {
                    'bbox': [int(x), int(y), int(width), int(height)],
                    'area': int(area),
                    'width': float(rod_width),
                    'length': float(rod_length),
                    'centroid': [float(centroids[label_id, 0]), float(centroids[label_id, 1])],
                }
                detection['left_rod'] = rod_info
                left_rods.append(rod_info)

            elif x + width > w - right_margin:
                rod_width = width
                rod_length = height
                rod_info = {
                    'bbox': [int(x), int(y), int(width), int(height)],
                    'area': int(area),
                    'width': float(rod_width),
                    'length': float(rod_length),
                    'centroid': [float(centroids[label_id, 0]), float(centroids[label_id, 1])],
                }
                detection['right_rod'] = rod_info
                right_rods.append(rod_info)

        all_detections.append(detection)

    total_images = len(all_detections)
    images_with_left = sum(1 for d in all_detections if d['left_rod'] is not None)
    images_with_right = sum(1 for d in all_detections if d['right_rod'] is not None)
    images_with_both = sum(
        1 for d in all_detections 
        if d['left_rod'] is not None and d['right_rod'] is not None
    )
    images_with_any = sum(
        1 for d in all_detections 
        if d['left_rod'] is not None or d['right_rod'] is not None
    )

    left_stats = {}
    if left_rods:
        left_areas = [r['area'] for r in left_rods]
        left_widths = [r['width'] for r in left_rods]
        left_lengths = [r['length'] for r in left_rods]

        left_stats = {
            'count': len(left_rods),
            'detection_rate': images_with_left / total_images if total_images > 0 else 0,
            'area': {
                'mean': float(np.mean(left_areas)),
                'std': float(np.std(left_areas)),
                'min': float(np.min(left_areas)),
                'max': float(np.max(left_areas)),
                'median': float(np.median(left_areas)),
            },
            'width': {
                'mean': float(np.mean(left_widths)),
                'std': float(np.std(left_widths)),
                'min': float(np.min(left_widths)),
                'max': float(np.max(left_widths)),
                'median': float(np.median(left_widths)),
            },
            'length': {
                'mean': float(np.mean(left_lengths)),
                'std': float(np.std(left_lengths)),
                'min': float(np.min(left_lengths)),
                'max': float(np.max(left_lengths)),
                'median': float(np.median(left_lengths)),
            },
        }

    right_stats = {}
    if right_rods:
        right_areas = [r['area'] for r in right_rods]
        right_widths = [r['width'] for r in right_rods]
        right_lengths = [r['length'] for r in right_rods]

        right_stats = {
            'count': len(right_rods),
            'detection_rate': images_with_right / total_images if total_images > 0 else 0,
            'area': {
                'mean': float(np.mean(right_areas)),
                'std': float(np.std(right_areas)),
                'min': float(np.min(right_areas)),
                'max': float(np.max(right_areas)),
                'median': float(np.median(right_areas)),
            },
            'width': {
                'mean': float(np.mean(right_widths)),
                'std': float(np.std(right_widths)),
                'min': float(np.min(right_widths)),
                'max': float(np.max(right_widths)),
                'median': float(np.median(right_widths)),
            },
            'length': {
                'mean': float(np.mean(right_lengths)),
                'std': float(np.std(right_lengths)),
                'min': float(np.min(right_lengths)),
                'max': float(np.max(right_lengths)),
                'median': float(np.median(right_lengths)),
            },
        }

    total_black_pixels = [d['total_black_pixels'] for d in all_detections]
    black_pixel_stats = {
        'mean': float(np.mean(total_black_pixels)),
        'std': float(np.std(total_black_pixels)),
        'min': int(np.min(total_black_pixels)),
        'max': int(np.max(total_black_pixels)),
        'median': float(np.median(total_black_pixels)),
    }

    results = {
        'dataset_name': dataset_name,
        'parameters': {
            'black_threshold': black_threshold,
            'min_area': min_area,
            'left_margin': left_margin,
            'right_margin': right_margin,
        },
        'summary': {
            'total_images': total_images,
            'images_with_left_rod': images_with_left,
            'images_with_right_rod': images_with_right,
            'images_with_both_rods': images_with_both,
            'images_with_any_rod': images_with_any,
            'left_rod_detection_rate': images_with_left / total_images if total_images > 0 else 0,
            'right_rod_detection_rate': images_with_right / total_images if total_images > 0 else 0,
            'both_rods_detection_rate': images_with_both / total_images if total_images > 0 else 0,
        },
        'left_rod_statistics': left_stats,
        'right_rod_statistics': right_stats,
        'black_pixel_statistics': black_pixel_stats,
        'detections': all_detections,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / f"ump_mask_analysis_{dataset_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到: {json_path}")

    print("\n" + "="*80)
    print("UMP杆黑色掩码分析报告")
    print("="*80)
    print(f"\n数据集: {dataset_name}")
    print(f"总图像数: {total_images}")
    print(f"\n检测参数:")
    print(f"  - 黑色阈值: {black_threshold}")
    print(f"  - 最小面积: {min_area} px²")
    print(f"  - 左边缘区域: {left_margin} px")
    print(f"  - 右边缘区域: {right_margin} px")

    print(f"\n检测率:")
    print(f"  - 检测到左UMP杆: {images_with_left}/{total_images} ({images_with_left/total_images*100:.1f}%)")
    print(f"  - 检测到右UMP杆: {images_with_right}/{total_images} ({images_with_right/total_images*100:.1f}%)")
    print(f"  - 同时检测到两个: {images_with_both}/{total_images} ({images_with_both/total_images*100:.1f}%)")
    print(f"  - 至少检测到一个: {images_with_any}/{total_images} ({images_with_any/total_images*100:.1f}%)")

    if left_stats:
        print(f"\n左UMP杆统计 (共 {left_stats['count']} 次检测):")
        print(f"  面积: {left_stats['area']['mean']:.1f} ± {left_stats['area']['std']:.1f} px²")
        print(f"        [最小: {left_stats['area']['min']:.0f}, 最大: {left_stats['area']['max']:.0f}, 中位数: {left_stats['area']['median']:.1f}]")
        print(f"  宽度: {left_stats['width']['mean']:.1f} ± {left_stats['width']['std']:.1f} px")
        print(f"        [最小: {left_stats['width']['min']:.0f}, 最大: {left_stats['width']['max']:.0f}, 中位数: {left_stats['width']['median']:.1f}]")
        print(f"  长度: {left_stats['length']['mean']:.1f} ± {left_stats['length']['std']:.1f} px")
        print(f"        [最小: {left_stats['length']['min']:.0f}, 最大: {left_stats['length']['max']:.0f}, 中位数: {left_stats['length']['median']:.1f}]")

    if right_stats:
        print(f"\n右UMP杆统计 (共 {right_stats['count']} 次检测):")
        print(f"  面积: {right_stats['area']['mean']:.1f} ± {right_stats['area']['std']:.1f} px²")
        print(f"        [最小: {right_stats['area']['min']:.0f}, 最大: {right_stats['area']['max']:.0f}, 中位数: {right_stats['area']['median']:.1f}]")
        print(f"  宽度: {right_stats['width']['mean']:.1f} ± {right_stats['width']['std']:.1f} px")
        print(f"        [最小: {right_stats['width']['min']:.0f}, 最大: {right_stats['width']['max']:.0f}, 中位数: {right_stats['width']['median']:.1f}]")
        print(f"  长度: {right_stats['length']['mean']:.1f} ± {right_stats['length']['std']:.1f} px")
        print(f"        [最小: {right_stats['length']['min']:.0f}, 最大: {right_stats['length']['max']:.0f}, 中位数: {right_stats['length']['median']:.1f}]")

    print(f"\n黑色像素统计:")
    print(f"  平均: {black_pixel_stats['mean']:.0f} ± {black_pixel_stats['std']:.0f} px")
    print(f"  [最小: {black_pixel_stats['min']}, 最大: {black_pixel_stats['max']}, 中位数: {black_pixel_stats['median']:.0f}]")

    print("="*80)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析所有图像的UMP杆黑色掩码")
    parser.add_argument("--dataset", type=str, default="robot_data_20260220_195724",
                       help="数据集名称")
    parser.add_argument("--black-threshold", type=int, default=30,
                       help="黑色检测阈值 (0-255)")
    parser.add_argument("--min-area", type=int, default=100,
                       help="最小连通域面积")
    parser.add_argument("--left-margin", type=int, default=50,
                       help="左边缘搜索区域宽度 (像素)")
    parser.add_argument("--right-margin", type=int, default=50,
                       help="右边缘搜索区域宽度 (像素)")
    parser.add_argument("--output-dir", type=str, 
                       default=str(OUTPUT_ROOT / "navigation"),
                       help="输出目录")

    args = parser.parse_args()

    analyze_all_images(
        dataset_name=args.dataset,
        black_threshold=args.black_threshold,
        min_area=args.min_area,
        left_margin=args.left_margin,
        right_margin=args.right_margin,
        output_dir=args.output_dir,
    )
