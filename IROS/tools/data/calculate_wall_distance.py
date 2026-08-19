#!/usr/bin/env python3

import os
import sys
import pandas as pd
from iros.paths import DATA_ROOT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_ALIGN_DIR = str(DATA_ROOT)

sys.path.insert(0, SCRIPT_DIR)
from tools.data.maze_io import (
    load_maze_params,
    load_maze_extractor,
    query_point_features,
    get_maze_shape
)

def add_d_wall_column(dataset_name):

    csv_path = os.path.join(MASK_ALIGN_DIR, dataset_name, 'tracking_results.csv')
    df = pd.read_csv(csv_path)
    print(f"加载了 {len(df)} 行数据")

    print("加载迷宫数据...")
    extractor = load_maze_extractor(dataset_name)
    maze_params = load_maze_params(dataset_name)
    maze_shape = get_maze_shape()

    if extractor is None:
        raise RuntimeError("无法加载 maze_features.pkl")
    if maze_params is None:
        raise RuntimeError("无法加载 maze_params.json")
    if maze_shape is None:
        raise RuntimeError("无法获取迷宫尺寸")

    target_name = maze_params.get('active_target', 'A')
    print(f"迷宫尺寸: {maze_shape}")
    print(f"目标: {target_name}")

    print("\n开始计算 d_wall...")
    d_wall_values = []
    for idx, row in df.iterrows():
        features = query_point_features(
            extractor, row['centroid_x'], row['centroid_y'], 
            maze_params, maze_shape, target_name
        )
        if features is not None:
            d_wall_values.append(features.get('d_wall', None))
        else:
            d_wall_values.append(None)

        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(df)} 行")

    df['d_wall'] = d_wall_values
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 已添加 d_wall 列到: {csv_path}")

    valid_d_wall = df['d_wall'].dropna()
    print(f"\n统计信息:")
    print(f"  有效数据: {len(valid_d_wall)}/{len(df)} ({len(valid_d_wall)/len(df)*100:.1f}%)")
    if len(valid_d_wall) > 0:
        print(f"  d_wall 均值: {valid_d_wall.mean():.2f}")
        print(f"  d_wall 中位数: {valid_d_wall.median():.2f}")
        print(f"  d_wall 最小值: {valid_d_wall.min():.2f}")
        print(f"  d_wall 最大值: {valid_d_wall.max():.2f}")
        print(f"  d_wall 标准差: {valid_d_wall.std():.2f}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Add wall-distance values to tracking_results.csv")
    parser.add_argument("dataset", help="Dataset name under IROS_DATA_ROOT")
    add_d_wall_column(parser.parse_args().dataset)
