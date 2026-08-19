# Trajectory CSV Export Guide

### Purpose

- **Description**: Documents how `outputs/navigation/trajectory_robot_data_20260220_195724_A.csv` and `..._B.csv` are generated from the raw data and navigation code.
- **Use cases**: Supports experiment reproduction, data inspection, and downstream analysis scripts, such as redrawing trajectories from CSV files.

### Data and Configuration Sources

- **Raw data directory**:
  - `data/navigation/robot_data_20260220_195724/`
  - It contains `tracking_results.csv`, `registration_output/maze_params.json`, `registration_output/maze_features.pkl`, and related files.

- **Loading entry point**:
  - `load_navigation_data(dataset_name, config)` in `src/iros/navigation/data_loader.py`:
    - Reads `tracking_results.csv` to obtain trajectory data.
    - Reads `maze_params.json` to obtain maze registration parameters.
    - Reads `maze_features.pkl` to obtain the `MazeFeatureExtractor`.
    - Uses fixed geometric parameters from `configs/navigation/default.json`, including `robot_radius_px`, `ump_rod_width_maze`, and `ump_rod_length_maze`, to construct `NavigationData`.

- **Navigation configuration**:
  - `src/iros/navigation/config.py` defines the `NavigationConfig` structure and the `load_config` function.
  - The default configuration file is `configs/navigation/default.json` under the project root. It contains:
    - Robot radius `robot_radius_px`.
    - UMP rod width and length `ump_rod_width_maze` / `ump_rod_length_maze`, when enabled.
    - Fixed target locations in maze coordinates under `targets_maze`, for example `{"A": (y, x), "B": (y, x)}`.

### Trajectory Generation Process

- **Full trajectory structure**:
  - In `src/iros/navigation/trajectory_generator.py`, `FullTrajectory` encapsulates the complete result of one offline path-planning run:
    - `waypoints`: each sampled A* path point corresponds to a `TrajectoryWaypoint` containing:
      - `center_maze`, `bottom_maze`, and `top_maze`: center and two endpoint positions in maze coordinates.
      - `center_cam`, `bottom_cam`, and `top_cam`: corresponding positions in camera coordinates.
      - `tangent_maze`: unit tangent vector `(dy, dx)` along the centerline.
    - `path_result`: the A* path-search result, including `path_maze`, `goal_maze`, and related fields.

- **Generation entry point**:
  - `generate_full_trajectory(data, config, target_name)`:
    - Starts from `NavigationData` and `NavigationConfig` and calls `path_planner` to plan the center path.
    - Computes the center, bottom, top, and tangent direction at each waypoint from the path and the robot/UMP geometric parameters.
    - Records the endpoint position for the selected target in `goal_maze`, which is later used to determine which rod endpoint is closer to the target.

### CSV Export Script

- **Location**: `src/iros/navigation/export_trajectory_csv.py`.

- **Core function**: `export_trajectory_to_csv(dataset_name, target_name, config_path=None, output_path=None)`:
  - Reuses the same logic as visualization through `_load_full_trajectory()`:
    - Locates the project root and loads `configs/navigation/default.json` as `NavigationConfig` by default.
    - Calls `load_navigation_data(dataset_name, config)` to load the requested dataset.
    - Calls `generate_full_trajectory(..., target_name=target_name)` to generate the full trajectory for the selected target.

- **Output path rules**:
  - If `output_path` is not specified:
    - Creates `outputs/navigation/` under the project root if it does not already exist.
    - Writes the file as `trajectory_<dataset_name>_<target_name>.csv`.

- **Command-line entry point**:
  - `main()` exposes the CLI through `argparse`:
    - `--dataset`: dataset name, default `robot_data_20260220_195724`.
    - `--target`: target name, default `A`; values such as `B` are also supported.
    - `--config`: optional custom configuration JSON path; if omitted, `configs/navigation/default.json` is used.
    - `--output` / `-o`: optional output CSV path; if omitted, the file is written to `outputs/navigation/trajectory_<dataset>_<target>.csv`.
  - Example commands from the project root:
    - `python -m iros.navigation.export_trajectory_csv --dataset robot_data_20260220_195724 --target A`
    - `python -m iros.navigation.export_trajectory_csv --dataset robot_data_20260220_195724 --target B`

### CSV Field Definitions

- **Basic columns**, all in maze coordinates and measured in pixels:
  - `index`: waypoint index starting from 0.
  - `center_y`, `center_x`: centerline position `(y, x)` in maze coordinates.
  - `bottom_y`, `bottom_x`: position of one rod endpoint in maze coordinates, typically corresponding to the lower endpoint or one arm.
  - `top_y`, `top_x`: position of the other rod endpoint in maze coordinates.

- **Near and far endpoints relative to the target**:
  - `closer_endpoint`: string value `"bottom"` or `"top"`, indicating which endpoint is closer to `goal_maze` according to Euclidean distance computed with `math.hypot`.
  - `closer_y`, `closer_x`: maze-coordinate position `(y, x)` of the closer endpoint.
  - `farther_endpoint`: label of the endpoint opposite to the closer endpoint.
  - `farther_y`, `farther_x`: maze-coordinate position `(y, x)` of the farther endpoint.

- **Tangent vectors**:
  - `tangent_closer_dy`, `tangent_closer_dx`: unit tangent vector `(dy, dx)` for the waypoint, taken from `TrajectoryWaypoint.tangent_maze` and assigned to the closer endpoint.
  - `tangent_farther_dy`, `tangent_farther_dx`: the same tangent vector assigned to the farther endpoint.
  - These four columns can be used to reconstruct orientation directly from the CSV, for example by drawing arrows indicating the direction of motion.

### Mapping to the A/B CSV Files

- **Target A CSV**:
  - Generated with: `python -m iros.navigation.export_trajectory_csv --dataset robot_data_20260220_195724 --target A`.
  - Output file: `outputs/navigation/trajectory_robot_data_20260220_195724_A.csv`.

- **Target B CSV**:
  - Generated with: `python -m iros.navigation.export_trajectory_csv --dataset robot_data_20260220_195724 --target B`.
  - Output file: `outputs/navigation/trajectory_robot_data_20260220_195724_B.csv`.

- **Consistency guarantee**:
  - Both CSV files use the same `NavigationConfig` and the same dataset; only `target_name` differs.
  - The trajectory-generation logic is identical to the visualization logic in `src/iros/navigation/visualize.py`; the result is simply exported to CSV for downstream analysis or plotting.
