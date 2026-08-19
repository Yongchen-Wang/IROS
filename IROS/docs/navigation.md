# Navigation module

The navigation package plans collision-aware trajectories for a rigid
microrobot controlled by two UMP arms. It covers data loading, coordinate
conversion, safety-cost construction, A* planning, endpoint geometry, arm-role
switching, trajectory export, and visualization.

## Data flow

```text
tracking_results.csv + maze_params.json + maze_features.pkl
                              │
                              ▼
                    load_navigation_data
                              │
                              ▼
                  build_safety_costmap
                              │
                              ▼
                  plan_best_path_from_region
                              │
                              ▼
                  generate_full_trajectory
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          CSV trajectory              visualization
```

Datasets live under `data/navigation/<dataset-name>` by default. Set
`IROS_DATA_ROOT` before starting Python to use another location.

## Configuration

Use `configs/navigation/default.json` as the starting point. The most important
fields are:

| Field | Meaning |
|---|---|
| `robot_radius_px` | Robot radius in image pixels |
| `ump_rod_width_maze` | UMP rod width in maze coordinates |
| `ump_rod_length_maze` | UMP rod length in maze coordinates |
| `start_region` | Circular or rectangular start region in camera coordinates |
| `targets_maze` | Named targets in maze `(y, x)` coordinates |
| `safety_margin_weight` | Penalty for paths close to obstacles |
| `path_smoothing_enabled` | Enable spline smoothing after A* |
| `max_arm_switches` | Maximum allowed master-arm changes |

## Commands

```bash
# Visualize one trajectory
python -m iros.navigation.visualize \
  --dataset <dataset-name> \
  --target A \
  --config configs/navigation/default.json \
  --no-show

# Export waypoints
python -m iros.navigation.export_trajectory_csv \
  --dataset <dataset-name> \
  --target A

# Create an animation
python -m iros.navigation.animate_path \
  --dataset <dataset-name> \
  --target A
```

Generated files default to `outputs/navigation`. Set `IROS_OUTPUT_ROOT` to
redirect them.

## Package map

| Module | Responsibility |
|---|---|
| `config.py` | Typed configuration and JSON serialization |
| `data_loader.py` | Dataset validation and loading |
| `coordinate_transform.py` | Camera/maze coordinate conversion |
| `costmap_builder.py` | Clearance-aware traversal costs |
| `path_planner.py` | Start sampling and A* search |
| `arm_switching.py` | Master/slave arm selection |
| `trajectory_generator.py` | Center and endpoint waypoint generation |
| `ump_geometry.py` | UMP rod masks and collision geometry |
| `export_trajectory_csv.py` | CSV export CLI |
| `visualize*.py`, `animate_path.py` | Static and animated output |
