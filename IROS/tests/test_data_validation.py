from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cv2")

from tools.data import import_wall_distance_fsr as importer


def _valid_tracking_frame(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "d_wall": np.linspace(1.0, 2.0, n),
            "iou": np.linspace(0.2, 0.8, n),
            "fsr_L_raw": np.arange(n, dtype=float),
            "fsr_R_raw": np.arange(n, dtype=float),
            "fsr_L_intent": np.linspace(0.0, 1.0, n),
            "fsr_R_intent": np.linspace(0.0, 1.0, n),
            "fsr_L_normalized": np.linspace(0.0, 1.0, n),
            "fsr_R_normalized": np.linspace(0.0, 1.0, n),
        }
    )


@pytest.mark.parametrize("column", importer.PIPELINE_NUMERIC_COLUMNS)
def test_validate_data_rejects_non_finite_numeric_columns(column: str) -> None:
    frame = _valid_tracking_frame()
    frame.loc[1, column] = np.nan

    with pytest.raises(ValueError, match=column):
        importer.validate_data(frame)


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf, "not-a-number"])
def test_validate_data_rejects_invalid_iou_values(bad_value: object) -> None:
    frame = _valid_tracking_frame()
    frame.loc[0, "iou"] = bad_value

    with pytest.raises(ValueError, match="iou"):
        importer.validate_data(frame)


def test_validate_data_accepts_finite_pipeline_data() -> None:
    assert importer.validate_data(_valid_tracking_frame()) is True


def test_update_alpha_labels_does_not_overwrite_on_invalid_input(
    tmp_path: Path,
) -> None:
    output = tmp_path / "alpha_labels.pkl"
    original = b"existing-label-file"
    output.write_bytes(original)
    frame = _valid_tracking_frame()
    frame.loc[0, "d_wall"] = np.nan

    with pytest.raises(ValueError, match="d_wall"):
        importer.update_alpha_labels(
            frame,
            tmp_path,
            {
                "baseline_L": 0.0,
                "override_L": 1.0,
                "baseline_R": 0.0,
                "override_R": 1.0,
            },
        )

    assert output.read_bytes() == original


def test_main_validates_before_writing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_name = "recording"
    recording_dir = tmp_path / dataset_name
    recording_dir.mkdir()
    csv_path = recording_dir / "tracking_results.csv"
    csv_path.write_text("centroid_x,centroid_y,iou\n1,2,0.5\n", encoding="utf-8")
    original_csv = csv_path.read_bytes()

    def add_invalid_wall_distance(frame: pd.DataFrame, _dataset: str) -> pd.DataFrame:
        frame["d_wall"] = np.nan
        return frame

    monkeypatch.setattr(importer, "MASK_ALIGN_DIR", str(tmp_path))
    monkeypatch.setattr(importer, "add_d_wall_column", add_invalid_wall_distance)
    monkeypatch.setattr(
        importer,
        "load_fsr_from_raw_csv",
        lambda *_args, **_kwargs: (np.array([0.5]), np.array([0.5])),
    )
    monkeypatch.setattr(
        importer,
        "resolve_fsr_calibration",
        lambda *_args, **_kwargs: {
            "baseline_L": 0.0,
            "override_L": 1.0,
            "baseline_R": 0.0,
            "override_R": 1.0,
        },
    )

    with pytest.raises(ValueError, match="d_wall"):
        importer.main(dataset_name)

    assert csv_path.read_bytes() == original_csv
    assert not (recording_dir / "alpha_labels.pkl").exists()
