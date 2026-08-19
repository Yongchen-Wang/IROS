# Training

Install the training dependencies from the repository root:

```bash
pip install -e '.[training]'
```

## Authority-negotiation network (Bi-CAST) and baselines

`training/core/train_model.py` trains Bi-CAST and the network-comparison
baselines from the paper (Table III): `three_layer_cnn`, `resnet18`,
`resnet50`, `tsm_resnet18`, `r2plus1d`, `mvit`, `vit`, `bicast`. Defaults
match the paper (AdamW, lr 1e-4, batch 32, 60 epochs, C = 5, f = 4,
λ_s = 0.1, λ_1 = 1e-3, w_L/w_R = 0.6/0.4); see
[docs/training_parameters.md](../docs/training_parameters.md).

Paper data split (Task 1 & 3 → train; Task 2 → val/test):

```bash
python -m training.core.train_model \
  --train_dirs data/task1_rec data/task3_rec \
  --valtest_dir data/task2_rec \
  --models bicast resnet18 resnet50
```

Each recording directory needs `images/` and `alpha_labels.pkl` with keys
`alpha_L`, `alpha_R` (merged annotator scores —
`tools/annotation/merge_alpha_annotations.py`), `fsr_L`, `fsr_R`, `iou`, and
`d_wall`.  `tools/data/import_wall_distance_fsr.py` now assembles the latter
four keys directly from `tracking_results.csv` plus the raw FSR stream.  To
match paper Eq. (4)-(5), provide the per-session natural-grip baseline and
deliberate-override force for both hands; recording-level min-max
normalisation is intentionally not used.  Example calibration file:

```json
{
  "left":  {"baseline": 1.2, "override": 4.8},
  "right": {"baseline": 1.1, "override": 4.6}
}
```

Place it as `<recording>/fsr_calibration.json` (or pass `--calibration`) and
run:

```bash
python -m tools.data.import_wall_distance_fsr task1_rec \
  --calibration /path/to/fsr_calibration.json
```

The script applies `I=(F-F_baseline)/(F_override-F_baseline)`, clips to
`[0,1]`, applies the paper's trailing `w=3` average, updates
`tracking_results.csv`, and creates/updates `alpha_labels.pkl` while
preserving any existing `alpha_L`/`alpha_R`.

Evaluate saved checkpoints on the test half of Task 2.  Table-III
smoothness is reported separately for the left and right arms:

```bash
python -m training.core.evaluate_models \
  --recording data/task2_rec --half second \
  --models bicast resnet18
```

Plot executed-vs-annotated authority curves (Eq. 8 aggregation):

```bash
python -m training.scripts.visualize_predictions \
  --checkpoint outputs/authority_models/bicast/model_bicast.pth \
  --recording data/task2_rec
```

## Table IV ablations

The paper's one-factor-at-a-time ablations are exposed in the actual Bi-CAST
model rather than substituted with standalone baselines:

- visual backbone: `resnet18`, `resnet34`, `resnet50`;
- input modality: `vision`, `vision_fsr`, `vision_fsr_safety`;
- smoothness: `--lambda_s 0` vs the default `0.1`;
- chunk size: `--chunk_size 1`, `5`, or `10` (changes both dataset targets and
  model heads).

Run the complete Table-IV sweep with:

```bash
python -m training.scripts.run_table_iv_ablations \
  --train_dirs data/task1_rec data/task3_rec \
  --valtest_dir data/task2_rec
```

The runner trains eight unique configurations (reusing the full Bi-CAST
reference row across groups) and writes `table_iv_results.json` in paper row
order.  Table-IV MAE/RMSE/R² are computed as the bilateral mean of the L/R
test metrics, matching the values reported in the paper.

Single ablations can also be run directly, e.g.:

```bash
python -m training.core.train_model \
  --train_dirs data/task1_rec data/task3_rec \
  --valtest_dir data/task2_rec --models bicast \
  --bicast_backbone resnet34 --input_modality vision_fsr --chunk_size 5
```

## Tracker (YOLOv8s-seg)

```bash
python -m training.scripts.train_yolo_seg --data path/to/dataset.yaml
```

## Environment check

```bash
python -m training.scripts.smoke_test
```
