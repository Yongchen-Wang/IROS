from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("torchvision")

from training.core import train_model


class StubAuthorityDataset:
    n_frames = 100
    calls: list[tuple[int, int] | None] = []

    def __init__(self, root, frame_range=None, **kwargs):
        del root, kwargs
        self.n = self.n_frames
        self.frame_range = frame_range
        self.calls.append(frame_range)


def test_temporal_split_returns_three_independent_ranges(monkeypatch) -> None:
    StubAuthorityDataset.calls = []
    monkeypatch.setattr(train_model, "AuthorityDataset", StubAuthorityDataset)

    train, val, test = train_model.temporal_split("unused", val_ratio=0.2)

    assert train is not val
    assert val is not test
    assert train is not test
    assert train.frame_range == (0, 60)
    assert val.frame_range == (60, 80)
    assert test.frame_range == (80, 100)
    assert StubAuthorityDataset.calls == [None, (0, 60), (60, 80), (80, 100)]


def test_temporal_split_accepts_an_explicit_test_ratio(monkeypatch) -> None:
    StubAuthorityDataset.calls = []
    monkeypatch.setattr(train_model, "AuthorityDataset", StubAuthorityDataset)

    train, val, test = train_model.temporal_split(
        "unused", val_ratio=0.1, test_ratio=0.3
    )

    assert train.frame_range == (0, 60)
    assert val.frame_range == (60, 70)
    assert test.frame_range == (70, 100)


@pytest.mark.parametrize(
    ("val_ratio", "test_ratio"),
    [
        (0.0, None),
        (-0.1, None),
        (1.0, None),
        (float("nan"), 0.1),
        (0.2, 0.0),
        (0.2, float("inf")),
        (0.4, 0.6),
    ],
)
def test_temporal_split_rejects_invalid_ratios(
    monkeypatch, val_ratio: float, test_ratio: float | None
) -> None:
    monkeypatch.setattr(train_model, "AuthorityDataset", StubAuthorityDataset)

    with pytest.raises(ValueError):
        train_model.temporal_split(
            "unused", val_ratio=val_ratio, test_ratio=test_ratio
        )


def test_temporal_split_rejects_empty_sample_partitions(monkeypatch) -> None:
    monkeypatch.setattr(StubAuthorityDataset, "n_frames", 30)
    monkeypatch.setattr(train_model, "AuthorityDataset", StubAuthorityDataset)

    with pytest.raises(ValueError, match="each split needs at least 8 frames"):
        train_model.temporal_split("unused", val_ratio=0.2)


def test_authority_dataset_keeps_windows_inside_frame_range(tmp_path: Path) -> None:
    n_frames = 40
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for frame in range(n_frames):
        (image_dir / f"{frame:05d}.jpg").touch()

    labels = {
        "alpha_L": np.zeros(n_frames, dtype=np.float32),
        "alpha_R": np.zeros(n_frames, dtype=np.float32),
        "fsr_L": np.zeros(n_frames, dtype=np.float32),
        "fsr_R": np.zeros(n_frames, dtype=np.float32),
        "iou": np.ones(n_frames, dtype=np.float32),
        "d_wall": np.ones(n_frames, dtype=np.float32),
    }
    with open(tmp_path / "alpha_labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    dataset = train_model.AuthorityDataset(
        tmp_path, frame_range=(10, 20), chunk_size=5
    )

    assert dataset.indices == [13, 14, 15]
    for target_start in dataset.indices:
        input_frames = range(target_start - train_model.WINDOW + 1, target_start + 1)
        target_frames = range(target_start, target_start + dataset.chunk_size)
        assert min(input_frames) >= 10
        assert max(target_frames) < 20
