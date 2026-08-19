import os
from pathlib import Path

def check_label_format(line):

    parts = line.strip().split()
    if len(parts) < 4:                                
        return False, f"坐标点太少（至少需要3个点，6个坐标值）"

    try:
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return False, "坐标格式错误（无法转换为数字）"

    if len(coords) % 2 != 0:
        return False, f"坐标数量不是偶数（x, y应该成对出现）"

    for coord in coords:
        if coord < 0 or coord > 1:
            return False, f"坐标超出范围 [0, 1]: {coord}"

    num_points = len(coords) // 2
    if num_points < 3:
        return False, f"点数太少（至少需要3个点，当前{num_points}个）"

    return True, "OK"

def check_and_fix_labels(labels_dir):

    if not os.path.exists(labels_dir):
        return 0, 0, 0

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    label_files.sort()

    total_count = 0
    fixed_count = 0
    error_count = 0
    error_details = []

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        total_count += 1

        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) == 0:
            continue

        seen = set()
        unique_lines = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)

        valid_lines = []
        for line in unique_lines:
            is_valid, error_msg = check_label_format(line)
            if is_valid:
                valid_lines.append(line)
            else:
                error_details.append(f"{label_file}: {error_msg} - {line[:50]}...")

        if len(valid_lines) != len(lines) or len(valid_lines) != len(unique_lines):
            if len(valid_lines) == 0:

                error_count += 1
                error_details.append(f"{label_file}: 所有标注都无效，文件可能为空")

            else:
                with open(label_path, 'w', encoding='utf-8') as f:
                    for line in valid_lines:
                        f.write(line + '\n')
                fixed_count += 1
                if len(lines) != len(valid_lines):
                    print(f"修复: {label_file} ({len(lines)} -> {len(valid_lines)} 行)")

    return total_count, fixed_count, error_count, error_details

if __name__ == "__main__":
    base_dir = "./yolo_dataset"

    print("=" * 60)
    print("检查 yolo_dataset 目录下所有 labels 文件")
    print("=" * 60)

    total_files_all = 0
    fixed_files_all = 0
    error_files_all = 0
    all_errors = []

    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        labels_dir = os.path.join(subdir_path, "labels")
        if not os.path.exists(labels_dir):
            continue

        print(f"\n检查目录: {labels_dir}")
        total, fixed, errors, error_details = check_and_fix_labels(labels_dir)

        total_files_all += total
        fixed_files_all += fixed
        error_files_all += errors
        all_errors.extend(error_details)

        print(f"  总文件数: {total}")
        print(f"  修复文件数: {fixed}")
        if errors > 0:
            print(f"  错误文件数: {errors}")

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"总文件数: {total_files_all}")
    print(f"修复文件数: {fixed_files_all}")
    print(f"错误文件数: {error_files_all}")

    if all_errors:
        print(f"\n发现 {len(all_errors)} 个错误/警告:")
        for error in all_errors[:20]:           
            print(f"  - {error}")
        if len(all_errors) > 20:
            print(f"  ... 还有 {len(all_errors) - 20} 个错误未显示")
