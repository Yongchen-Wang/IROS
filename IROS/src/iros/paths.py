from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
ASSET_ROOT = PROJECT_ROOT / "assets"
PROJECT_DATA_ROOT = PROJECT_ROOT / "data"
DATA_ROOT = Path(os.environ.get("IROS_DATA_ROOT", PROJECT_DATA_ROOT / "navigation"))
RAW_DATA_ROOT = Path(os.environ.get("IROS_RAW_DATA_ROOT", PROJECT_DATA_ROOT / "raw"))
FRAME_ROOT = Path(os.environ.get("IROS_FRAME_ROOT", PROJECT_DATA_ROOT / "frames"))
VIDEO_ROOT = Path(os.environ.get("IROS_VIDEO_ROOT", PROJECT_DATA_ROOT / "videos"))
CHECKPOINT_ROOT = Path(os.environ.get("IROS_CHECKPOINT_ROOT", PROJECT_DATA_ROOT / "checkpoints"))
TRAINING_DATA_ROOT = Path(
    os.environ.get("IROS_TRAINING_DATA_ROOT", PROJECT_DATA_ROOT / "training")
)
SHARED_CONTROL_DATA_ROOT = Path(
    os.environ.get("IROS_SHARED_CONTROL_DATA_ROOT", PROJECT_DATA_ROOT / "shared_control")
)
OUTPUT_ROOT = Path(os.environ.get("IROS_OUTPUT_ROOT", PROJECT_ROOT / "outputs"))
