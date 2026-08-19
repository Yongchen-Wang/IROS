import os
import shutil
from pathlib import Path

def merge_yolo_datasets(source_dir, output_dir):

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    subdirs = [d for d in os.listdir(source_dir) 
               if os.path.isdir(os.path.join(source_dir, d)) 
               and d not in ['images', 'labels']]             

    print(f"找到 {len(subdirs)} 个子目录: {subdirs}\n")

    total_images = 0
    total_labels = 0
    frame_counter = 0

    for subdir in subdirs:
        subdir_path = os.path.join(source_dir, subdir)
        sub_images_dir = os.path.join(subdir_path, "images")
        sub_labels_dir = os.path.join(subdir_path, "labels")

        if not os.path.exists(sub_images_dir) or not os.path.exists(sub_labels_dir):
            print(f"跳过 {subdir}: 缺少 images 或 labels 目录")
            continue

        image_files = [f for f in os.listdir(sub_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()

        print(f"处理 {subdir}: {len(image_files)} 个文件")

        for img_file in image_files:

            new_name = f"{frame_counter:05d}{os.path.splitext(img_file)[1]}"

            src_img = os.path.join(sub_images_dir, img_file)
            dst_img = os.path.join(images_dir, new_name)
            shutil.copy2(src_img, dst_img)

            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label = os.path.join(sub_labels_dir, label_file)
            if os.path.exists(src_label):
                new_label_name = f"{frame_counter:05d}.txt"
                dst_label = os.path.join(labels_dir, new_label_name)
                shutil.copy2(src_label, dst_label)
                total_labels += 1
            else:
                print(f"  警告: {subdir}/{label_file} 不存在")

            total_images += 1
            frame_counter += 1

        print(f"  [OK] 完成: {len(image_files)} 个文件\n")

    print("=" * 60)
    print("合并完成！")
    print("=" * 60)
    print(f"总图像数: {total_images}")
    print(f"总标注数: {total_labels}")
    print(f"输出目录: {output_dir}")
    print(f"  - 图像: {images_dir}")
    print(f"  - 标注: {labels_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge child YOLO datasets into one dataset")
    parser.add_argument("source", help="Directory containing child datasets")
    parser.add_argument("output", help="Merged output directory")
    args = parser.parse_args()
    source_dir = args.source
    output_dir = args.output

    print("=" * 60)
    print("合并 YOLO 数据集")
    print("=" * 60)
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}\n")

    merge_yolo_datasets(source_dir, output_dir)
