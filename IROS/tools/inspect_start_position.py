#!/usr/bin/env python3

from pathlib import Path

import numpy as np

from iros.navigation.config import load_config
from iros.navigation.data_loader import load_navigation_data
from iros.navigation.coordinate_transform import cam_to_maze, maze_to_cam
from iros.navigation.costmap_builder import build_safety_costmap
from iros.navigation.path_planner import (
    _sample_start_points_in_camera,
    _nearest_valid_cell,
)
from iros.paths import CONFIG_ROOT

def main(
    dataset_name: str = "robot_data_20260220_195724",
    config_path: str | None = None,
) -> None:

    config_file = (
        Path(config_path)
        if config_path is not None
        else CONFIG_ROOT / "navigation" / "example.json"
    )
    config = load_config(str(config_file))

    print("=" * 60)
    print(f"检查起点坐标  数据集: {dataset_name}")
    print("=" * 60)

    data = load_navigation_data(dataset_name, config)

    maze_shape = tuple(data.maze_extractor.free_mask.shape)
    maze_params = data.maze_params

    if hasattr(data.maze_extractor, "start_point"):
        start_y, start_x = data.maze_extractor.start_point
        print("\n[1] 数据集中的 start_point")
        print(f"    maze 坐标: (y = {start_y:.2f}, x = {start_x:.2f})")

        cam_x, cam_y = maze_to_cam(start_y, start_x, maze_params, maze_shape)
        print(f"    对应 camera 坐标: (x = {cam_x:.2f}, y = {cam_y:.2f})")
    else:
        print("\n[1] 数据集不包含 maze_extractor.start_point")

    if config.start_region is not None and hasattr(config.start_region, "center"):
        cam_cx, cam_cy = config.start_region.center
        radius = getattr(config.start_region, "radius", 0.0)
        print("\n[2] 配置中的 start_region")
        print(f"    camera center: (x = {cam_cx:.2f}, y = {cam_cy:.2f}), radius = {radius:.2f}")

        maze_y, maze_x = cam_to_maze(
            cam_x=float(cam_cx),
            cam_y=float(cam_cy),
            maze_params=maze_params,
            maze_shape=maze_shape,
        )
        print(f"    对应 maze 坐标: (y = {maze_y:.2f}, x = {maze_x:.2f})")
    else:
        print("\n[2] 配置中的 start_region: None（将使用数据集 start_point 作为起点）")

    print("\n[3] 实际采样并投影到 costmap 后的起点")

    costmap = build_safety_costmap(
        maze_extractor=data.maze_extractor,
        robot_radius_px=data.robot_radius_px,
        config=config,
        ump_rod_mask=None,
    )

    rng = np.random.default_rng(42)

    if config.start_region is not None and hasattr(config.start_region, "center"):
        cam_samples = _sample_start_points_in_camera(
            start_region=config.start_region,
            n_samples=config.n_start_samples,
            rng=rng,
        )

        for i, (cam_x, cam_y) in enumerate(cam_samples):
            maze_y, maze_x = cam_to_maze(
                cam_x=float(cam_x),
                cam_y=float(cam_y),
                maze_params=maze_params,
                maze_shape=maze_shape,
            )
            cell = _nearest_valid_cell(costmap, maze_y, maze_x)
            print(f"    样本 {i+1}:")
            print(f"      camera: (x = {cam_x:.2f}, y = {cam_y:.2f})")
            print(f"      maze  : (y = {maze_y:.2f}, x = {maze_x:.2f})")
            print(f"      最近可行 cell 索引: {cell}")
    else:

        if hasattr(data.maze_extractor, "start_point"):
            start_y, start_x = data.maze_extractor.start_point
            cam_x, cam_y = maze_to_cam(start_y, start_x, maze_params, maze_shape)
            print("    未设置 start_region，使用数据集中的 start_point：")
            print(f"      maze 起点: (y = {start_y:.2f}, x = {start_x:.2f})")
            print(f"      camera 起点: (x = {cam_x:.2f}, y = {cam_y:.2f})")
        else:
            print("    既没有 start_region 也没有 start_point，无法确定起点。")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="检查导航起点坐标")
    parser.add_argument(
        "--dataset",
        type=str,
        default="robot_data_20260220_195724",
        help="数据集名称",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（可选）",
    )

    args = parser.parse_args()
    main(dataset_name=args.dataset, config_path=args.config)
