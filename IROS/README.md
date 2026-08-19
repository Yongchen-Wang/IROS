# IROS

IROS is a compact research codebase for dual-UMP microrobot navigation,
shared-control learning, maze analysis, and data annotation.

## Project layout

```text
IROS/
├── src/iros/                 # Installable Python package
│   ├── navigation/           # Planning, trajectories, and visualization
│   ├── navigation_experiments/
│   ├── maze/                 # Maze feature extraction
│   └── models/               # Bi-CAST authority-negotiation network
├── configs/navigation/       # Navigation JSON configurations
├── examples/                 # Small runnable examples
├── tools/                    # Annotation and data tools
├── training/                 # Training code and orchestration scripts
├── docs/                     # Focused technical documentation
├── data/                     # Local data (ignored by Git)
└── outputs/                  # Generated results (ignored by Git)
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install optional dependencies only for the features you need:

```bash
pip install -e '.[vision]'    # OpenCV and maze feature extraction
pip install -e '.[training]'  # PyTorch model training
pip install -e '.[ui]'        # Gradio annotation tools
pip install -e '.[dev]'       # Tests and linting
```

## Data and output paths

Navigation datasets are read from `data/navigation/<dataset-name>` by default.
Each dataset is expected to contain:

```text
<dataset-name>/
├── tracking_results.csv
└── registration_output/
    ├── maze_params.json
    └── maze_features.pkl
```

Keep large data outside the repository by setting `IROS_DATA_ROOT` to the
directory that contains the navigation datasets. Set `IROS_OUTPUT_ROOT` to
redirect generated files. For example:

```bash
export IROS_DATA_ROOT=/path/to/navigation-datasets
export IROS_OUTPUT_ROOT=/path/to/results
```

### Paper-aligned data preparation

Tracking videos are subsampled every other frame by default (`--interval 2`),
matching Sec. IV-B.  Authority FSR preprocessing uses the per-session natural
grip / deliberate-override calibration from Eq. (4), followed by the `w=3`
trailing average from Eq. (5); the importer also assembles
`fsr_L/fsr_R/iou/d_wall` into `alpha_labels.pkl`.  See
[training/README.md](training/README.md) for the exact commands.

## Common commands

```bash
# Export a planned trajectory
iros-export-trajectory \
  --dataset robot_data_20260220_195724 \
  --target A \
  --config configs/navigation/default.json

# Visualize a trajectory without opening a window
iros-visualize \
  --dataset robot_data_20260220_195724 \
  --target A \
  --config configs/navigation/default.json \
  --no-show

# Run the lightweight test suite
pytest
```

See [navigation.md](docs/navigation.md),
[trajectory_export.md](docs/trajectory_export.md),
[bicast_model.md](docs/bicast_model.md), and
[training_parameters.md](docs/training_parameters.md) for subsystem details.


