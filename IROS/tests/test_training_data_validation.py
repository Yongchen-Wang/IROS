from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("torchvision")

from training.core import train_model


def _write_recording(root: Path, labels: dict[str, np.ndarray], n_images: int = 8) -> None:
    image_dir = root / "images"
    image_dir.mkdir()
    for frame in range(n_images):
        (image_dir / f"{frame:05d}.jpg").touch()
    with open(root / "alpha_labels.pkl", "wb") as stream:
        pickle.dump(labels, stream)


def _valid_labels(n: int = 8) -> dict[str, np.ndarray]:
    return {
        "alpha_L": np.zeros(n, dtype=np.float32),
        "alpha_R": np.zeros(n, dtype=np.float32),
        "fsr_L": np.zeros(n, dtype=np.float32),
        "fsr_R": np.zeros(n, dtype=np.float32),
        "iou": np.ones(n, dtype=np.float32),
        "d_wall": np.ones(n, dtype=np.float32),
    }


@pytest.mark.parametrize("key", train_model.REQUIRED_LABEL_KEYS)
def test_authority_dataset_rejects_non_finite_training_arrays(
    tmp_path: Path, key: str
) -> None:
    labels = _valid_labels(n=9)
    labels[key][-1] = np.nan
    _write_recording(tmp_path, labels, n_images=8)

    with pytest.raises(ValueError, match=key):
        train_model.AuthorityDataset(tmp_path)


def test_authority_dataset_rejects_float32_overflow(tmp_path: Path) -> None:
    labels = _valid_labels()
    labels["d_wall"] = np.full(8, 1e100, dtype=np.float64)
    _write_recording(tmp_path, labels)

    with pytest.raises(ValueError, match="d_wall"):
        train_model.AuthorityDataset(tmp_path)


def test_zero_wall_distance_normalizes_without_nan(tmp_path: Path) -> None:
    labels = _valid_labels()
    labels["d_wall"][:] = 0.0
    _write_recording(tmp_path, labels)

    dataset = train_model.AuthorityDataset(tmp_path)

    assert np.isfinite(dataset.safety).all()
    np.testing.assert_array_equal(dataset.safety[:, 1], np.zeros(8))
