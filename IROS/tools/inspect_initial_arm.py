#!/usr/bin/env python3

from iros.navigation.arm_switching import ArmRole
from iros.navigation.config import load_config
from iros.navigation.data_loader import load_navigation_data
from iros.navigation.trajectory_generator import generate_full_trajectory
from iros.paths import CONFIG_ROOT

def check_initial_arm_position(
    dataset_name: str = "robot_data_20260220_195724",
    target_name: str = "A",
    config_path: str = None,
) -> None:

    config = load_config(config_path or str(CONFIG_ROOT / "navigation" / "example.json"))

    print("=" * 60)
    print(f"检查数据集: {dataset_name}, 目标: {target_name}")
    print("=" * 60)

    print(f"\n加载数据集...")
    data = load_navigation_data(dataset_name, config)

    print(f"生成轨迹...")
    traj = generate_full_trajectory(data, config, target_name=target_name)

    initial_wp = traj.waypoints[0]

    goal_maze = traj.path_result.goal_maze

    import numpy as np
    bottom_pt = np.array(initial_wp.bottom_maze)
    top_pt = np.array(initial_wp.top_maze)
    goal_pt = np.array(goal_maze)

    bottom_dist = np.hypot(bottom_pt[0] - goal_pt[0], bottom_pt[1] - goal_pt[1])
    top_dist = np.hypot(top_pt[0] - goal_pt[0], top_pt[1] - goal_pt[1])

    print(f"\n初始状态信息:")
    print(f"  起始中心位置 (maze): ({initial_wp.center_maze[0]:.2f}, {initial_wp.center_maze[1]:.2f})")
    print(f"  目标位置 (maze): ({goal_pt[0]:.2f}, {goal_pt[1]:.2f})")
    print(f"\n端点位置:")
    print(f"  左臂 (bottom) 位置: ({bottom_pt[0]:.2f}, {bottom_pt[1]:.2f})")
    print(f"  右臂 (top) 位置: ({top_pt[0]:.2f}, {top_pt[1]:.2f})")
    print(f"\n到目标的距离:")
    print(f"  左臂到目标距离: {bottom_dist:.2f} pixels")
    print(f"  右臂到目标距离: {top_dist:.2f} pixels")

    initial_master = initial_wp.master
    print(f"\n初始主导臂 (Master): {initial_master.value}")

    if initial_master == ArmRole.BOTTOM:
        print(f"  → 左臂在前 (Bottom/左臂是主导臂)")
    else:
        print(f"  → 右臂在前 (Top/右臂是主导臂)")

    if traj.switch_indices:
        print(f"\n轨迹中有 {len(traj.switch_indices)} 次臂切换:")
        for switch_idx in traj.switch_indices:
            if switch_idx < len(traj.waypoints):
                switch_wp = traj.waypoints[switch_idx]
                print(f"  在waypoint {switch_idx}: 切换到 {switch_wp.master.value}")
    else:
        print(f"\n轨迹中没有臂切换（全程保持初始主导臂）")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="检查轨迹的初始状态")
    parser.add_argument("--dataset", type=str, default="robot_data_20260220_195724",
                       help="数据集名称")
    parser.add_argument("--target", type=str, default="A",
                       help="目标名称 (A, B, C等)")
    parser.add_argument("--config", type=str, default=None,
                       help="配置文件路径（可选）")

    args = parser.parse_args()

    check_initial_arm_position(
        dataset_name=args.dataset,
        target_name=args.target,
        config_path=args.config,
    )
