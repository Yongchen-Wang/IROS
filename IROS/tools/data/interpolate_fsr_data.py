#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from iros.paths import DATA_ROOT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_ALIGN_DIR = str(DATA_ROOT)

def find_original_fsr_data(df):

    fsr_L = df['fsr_L_raw'].values
    fsr_R = df['fsr_R_raw'].values

    if len(fsr_L) < 2:
        return 0, len(fsr_L)

    last_valid_idx = None
    for i in range(len(fsr_L) - 1, -1, -1):
        if pd.notna(fsr_L[i]) and pd.notna(fsr_R[i]):
            last_valid_idx = i
            break

    if last_valid_idx is None:
        return 0, len(fsr_L)

    check_window = min(10, len(fsr_L) // 10)
    if check_window > 0:
        last_values_L = fsr_L[-check_window:]
        last_values_R = fsr_R[-check_window:]

        if (len(np.unique(last_values_L)) == 1 and 
            len(np.unique(last_values_R)) == 1):

            fill_value_L = last_values_L[0]
            fill_value_R = last_values_R[0]

            original_end = len(fsr_L)
            for i in range(len(fsr_L) - check_window - 1, -1, -1):
                if fsr_L[i] != fill_value_L or fsr_R[i] != fill_value_R:
                    original_end = i + 1
                    break

            return 0, original_end

    return 0, last_valid_idx + 1

def redistribute_fsr_with_interpolation(df):

    start_idx, end_idx = find_original_fsr_data(df)
    n_original = end_idx - start_idx
    n_total = len(df)

    print(f"原始FSR数据范围: [{start_idx}, {end_idx}), 共 {n_original} 个数据点")
    print(f"总数据长度: {n_total}")

    if n_original >= n_total:
        print("原始数据已经足够，无需重新分布")
        return df

    original_fsr_L = df['fsr_L_raw'].iloc[start_idx:end_idx].values
    original_fsr_R = df['fsr_R_raw'].iloc[start_idx:end_idx].values

    original_indices = np.linspace(0, n_total - 1, n_original)
    target_indices = np.arange(n_total)

    interp_func_L = interp1d(
        original_indices, 
        original_fsr_L, 
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )
    interp_func_R = interp1d(
        original_indices,
        original_fsr_R,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )

    new_fsr_L = interp_func_L(target_indices)
    new_fsr_R = interp_func_R(target_indices)

    new_fsr_L = np.round(new_fsr_L).astype(int)
    new_fsr_R = np.round(new_fsr_R).astype(int)

    df['fsr_L_raw'] = new_fsr_L
    df['fsr_R_raw'] = new_fsr_R

    print(f"✅ 已重新分布FSR数据")
    print(f"   FSR L 范围: [{new_fsr_L.min()}, {new_fsr_L.max()}]")
    print(f"   FSR R 范围: [{new_fsr_R.min()}, {new_fsr_R.max()}]")

    return df

def normalize_fsr(df):

    raise RuntimeError(
        "Min-max FSR normalisation has been removed from the paper-aligned "
        "pipeline. Run `python -m tools.data.import_wall_distance_fsr ...` "
        "with baseline/override calibration instead."
    )

def process_dataset(dataset_name):

    csv_path = os.path.join(MASK_ALIGN_DIR, dataset_name, 'tracking_results.csv')

    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        return False

    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*60}")

    print(f"\n加载 tracking_results.csv: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ 加载了 {len(df)} 行数据")

    if 'fsr_L_raw' not in df.columns or 'fsr_R_raw' not in df.columns:
        print("❌ CSV文件缺少FSR列")
        return False

    df = redistribute_fsr_with_interpolation(df)

    print("提示: 该脚本只做 raw FSR 插值；论文归一化请运行 import_wall_distance_fsr.py")

    df.to_csv(csv_path, index=False)
    print(f"\n✅ 已保存结果到: {csv_path}")

    return True

def main(dataset_names):

    success_count = 0
    for dataset_name in dataset_names:
        if process_dataset(dataset_name):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"处理完成！成功处理 {success_count}/{len(dataset_names)} 个数据集")
    print(f"{'='*60}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Interpolate FSR values in navigation datasets")
    parser.add_argument("datasets", nargs="+", help="Dataset names under IROS_DATA_ROOT")
    main(parser.parse_args().datasets)
