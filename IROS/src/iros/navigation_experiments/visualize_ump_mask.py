import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .ump_rod_detector import detect_ump_rods_from_images
from ..paths import DATA_ROOT, OUTPUT_ROOT

def visualize_ump_mask(
    dataset_name: str,
    black_threshold: int = 30,
    min_area: int = 100,
    left_margin: int = 50,
    right_margin: int = 50,
    max_images: int = 10,
    output_path: str = str(OUTPUT_ROOT / "navigation" / "ump_mask_visualization.png"),
):

    dataset_dir = DATA_ROOT / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    print(f"Detecting UMP rods from dataset: {dataset_name}")
    detections = detect_ump_rods_from_images(
        dataset_dir=dataset_dir,
        black_threshold=black_threshold,
        min_area=min_area,
        left_margin=left_margin,
        right_margin=right_margin,
        max_images=max_images,
    )

    if not detections:
        print("Warning: No UMP rods detected in any images!")
        print("Try adjusting parameters:")
        print(f"  - black_threshold: {black_threshold} (try lower, e.g., 20)")
        print(f"  - min_area: {min_area} (try lower, e.g., 50)")
        print(f"  - left_margin: {left_margin}")
        print(f"  - right_margin: {right_margin}")
        return

    print(f"Found UMP rods in {len(detections)} images")

    num_vis = min(len(detections), max_images)

    fig, axes = plt.subplots(num_vis, 4, figsize=(20, 5 * num_vis))
    if num_vis == 1:
        axes = axes.reshape(1, -1)

    for idx, detection in enumerate(detections[:num_vis]):
        img_path = detection['image_path']
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)

        axes[idx, 0].imshow(img_rgb)
        axes[idx, 0].set_title(f"Original Image\n{Path(img_path).name}")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(gray, cmap='gray')
        axes[idx, 1].set_title("Grayscale")
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(binary, cmap='gray')
        axes[idx, 2].set_title("Black Mask\n(Threshold < {})".format(black_threshold))
        axes[idx, 2].axis('off')

        img_overlay = img_rgb.copy()

        cv2.rectangle(img_overlay, (0, 0), (left_margin, h), (255, 255, 0), 2)

        cv2.rectangle(img_overlay, (w - right_margin, 0), (w, h), (255, 255, 0), 2)

        if detection['left_rod'] is not None:
            x, y, width, height = detection['left_rod']['bbox']
            cv2.rectangle(img_overlay, (x, y), (x + width, y + height), (0, 255, 0), 3)
            cv2.putText(img_overlay, f"Left: {detection['left_rod']['area']}px", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if detection['right_rod'] is not None:
            x, y, width, height = detection['right_rod']['bbox']
            cv2.rectangle(img_overlay, (x, y), (x + width, y + height), (255, 0, 0), 3)
            cv2.putText(img_overlay, f"Right: {detection['right_rod']['area']}px", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        axes[idx, 3].imshow(img_overlay)
        axes[idx, 3].set_title("Detection Overlay\n(Green=Left, Red=Right)")
        axes[idx, 3].axis('off')

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    print("\nDetection Statistics:")
    left_rods = [d['left_rod'] for d in detections if d['left_rod'] is not None]
    right_rods = [d['right_rod'] for d in detections if d['right_rod'] is not None]

    if left_rods:
        left_areas = [r['area'] for r in left_rods]
        left_widths = [r['width'] for r in left_rods]
        left_lengths = [r['length'] for r in left_rods]
        print(f"  Left rod: {len(left_rods)} detections")
        print(f"    - Area: {np.mean(left_areas):.1f} ± {np.std(left_areas):.1f} px")
        print(f"    - Width: {np.mean(left_widths):.1f} ± {np.std(left_widths):.1f} px")
        print(f"    - Length: {np.mean(left_lengths):.1f} ± {np.std(left_lengths):.1f} px")

    if right_rods:
        right_areas = [r['area'] for r in right_rods]
        right_widths = [r['width'] for r in right_rods]
        right_lengths = [r['length'] for r in right_rods]
        print(f"  Right rod: {len(right_rods)} detections")
        print(f"    - Area: {np.mean(right_areas):.1f} ± {np.std(right_areas):.1f} px")
        print(f"    - Width: {np.mean(right_widths):.1f} ± {np.std(right_widths):.1f} px")
        print(f"    - Length: {np.mean(right_lengths):.1f} ± {np.std(right_lengths):.1f} px")

    if not left_rods and not right_rods:
        print("  No rods detected!")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize UMP rod black mask detection")
    parser.add_argument("--dataset", type=str, default="robot_data_20260220_195724",
                       help="Dataset name")
    parser.add_argument("--black-threshold", type=int, default=30,
                       help="Black detection threshold (0-255)")
    parser.add_argument("--min-area", type=int, default=100,
                       help="Minimum connected component area")
    parser.add_argument("--left-margin", type=int, default=50,
                       help="Left edge search margin (pixels)")
    parser.add_argument("--right-margin", type=int, default=50,
                       help="Right edge search margin (pixels)")
    parser.add_argument("--max-images", type=int, default=10,
                       help="Maximum number of images to visualize")
    parser.add_argument("--output", type=str, 
                       default=str(OUTPUT_ROOT / "navigation" / "ump_mask_visualization.png"),
                       help="Output path for visualization")

    args = parser.parse_args()

    visualize_ump_mask(
        dataset_name=args.dataset,
        black_threshold=args.black_threshold,
        min_area=args.min_area,
        left_margin=args.left_margin,
        right_margin=args.right_margin,
        max_images=args.max_images,
        output_path=args.output,
    )
