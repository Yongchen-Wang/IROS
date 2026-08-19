from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import NavigationConfig
from ..paths import DATA_ROOT

@dataclass
class NavigationData:

    tracking_df: pd.DataFrame
    maze_params: Dict[str, Any]
    maze_extractor: Any
    robot_radius_px: float
    ump_rod_width_maze: Optional[float] = None
    ump_rod_length_maze: Optional[float] = None

def _dataset_dir(dataset_name: str) -> Path:

    if not dataset_name:
        raise ValueError("dataset_name must be a non-empty string")
    return DATA_ROOT / dataset_name

def load_tracking_results(dataset_name: str) -> pd.DataFrame:

    ds_dir = _dataset_dir(dataset_name)
    csv_path = ds_dir / "tracking_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"tracking_results.csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"tracking_results.csv is empty: {csv_path}")
    return df

def load_maze_params(dataset_name: str) -> Dict[str, Any]:

    ds_dir = _dataset_dir(dataset_name)
    params_path = ds_dir / "registration_output" / "maze_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"maze_params.json not found: {params_path}")
    with params_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_maze_extractor(dataset_name: str) -> Any:

    ds_dir = _dataset_dir(dataset_name)
    pkl_path = ds_dir / "registration_output" / "maze_features.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"maze_features.pkl not found: {pkl_path}")

    import sys

    from ..maze import features as maze_features

    sys.modules.setdefault("maze_feature_extractor_v3_final", maze_features)
    with pkl_path.open("rb") as f:
        extractor = pickle.load(f)
    return extractor

def load_navigation_data(dataset_name: str, config: Optional[NavigationConfig] = None) -> NavigationData:

    if config is None:
        raise ValueError("NavigationConfig must be provided for Navigation V1.")

    tracking_df = load_tracking_results(dataset_name)
    maze_params = load_maze_params(dataset_name)
    maze_extractor = load_maze_extractor(dataset_name)

    if config.robot_radius_px is None:
        raise ValueError(
            "NavigationConfig.robot_radius_px must be set for Navigation V1; "
            "automatic estimation from images/tracking is disabled."
        )
    robot_radius_px: float = float(config.robot_radius_px)
    print(f"Using explicit robot_radius_px from config (fixed model): {robot_radius_px:.2f}")

    ump_rod_width_maze: Optional[float] = None
    ump_rod_length_maze: Optional[float] = None

    if config.enable_ump_rod_detection:
        if config.ump_rod_width_maze is None or config.ump_rod_length_maze is None:
            raise ValueError(
                "UMP rod collision checking is enabled (enable_ump_rod_detection=True), "
                "but ump_rod_width_maze / ump_rod_length_maze are not fully specified "
                "in NavigationConfig."
            )
        ump_rod_width_maze = float(config.ump_rod_width_maze)
        ump_rod_length_maze = float(config.ump_rod_length_maze)
        print(
            f"Using fixed UMP rod dimensions from config: "
            f"width={ump_rod_width_maze:.2f}, length={ump_rod_length_maze:.2f} (maze pixels)"
        )

    return NavigationData(
        tracking_df=tracking_df,
        maze_params=maze_params,
        maze_extractor=maze_extractor,
        robot_radius_px=robot_radius_px,
        ump_rod_width_maze=ump_rod_width_maze,
        ump_rod_length_maze=ump_rod_length_maze,
    )
