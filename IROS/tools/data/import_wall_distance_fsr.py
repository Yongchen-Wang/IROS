#!/usr/bin/env python3

import os
import sys
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tools.data.interpolate_fsr_data import redistribute_fsr_with_interpolation
from iros.paths import DATA_ROOT, RAW_DATA_ROOT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_ALIGN_DIR = str(DATA_ROOT)

DEFAULT_DATASET_NAME = 'robot_data_20260301_162357'

PIPELINE_NUMERIC_COLUMNS = (
    "d_wall",
    "iou",
    "fsr_L_raw",
    "fsr_R_raw",
    "fsr_L_intent",
    "fsr_R_intent",
    "fsr_L_normalized",
    "fsr_R_normalized",
)
TRAINING_LABEL_KEYS = ("alpha_L", "alpha_R", "fsr_L", "fsr_R", "iou", "d_wall")

sys.path.insert(0, SCRIPT_DIR)
from tools.data.maze_io import (
    load_maze_params,
    load_maze_extractor,
    query_point_features,
    get_maze_shape
)

def add_d_wall_column(df, dataset_name):

    print("加载迷宫数据...")
    extractor = load_maze_extractor(dataset_name)
    maze_params = load_maze_params(dataset_name)
    maze_shape = get_maze_shape()

    if extractor is None:
        raise RuntimeError("无法加载 maze_features.pkl")
    if maze_params is None:
        raise RuntimeError("无法加载 maze_params.json")
    if maze_shape is None:
        raise RuntimeError("无法获取迷宫尺寸")

    target_name = maze_params.get('active_target', 'A')
    print(f"迷宫尺寸: {maze_shape}")
    print(f"目标: {target_name}")

    print("\n开始计算 d_wall...")
    d_wall_values = []
    for idx, row in df.iterrows():
        features = query_point_features(
            extractor, row['centroid_x'], row['centroid_y'], 
            maze_params, maze_shape, target_name
        )
        if features is not None:
            d_wall_values.append(features.get('d_wall', None))
        else:
            d_wall_values.append(None)

        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(df)} 行")

    df['d_wall'] = d_wall_values
    print(f"✅ 已添加 d_wall 列")

    valid_d_wall = df['d_wall'].dropna()
    print(f"\nd_wall 统计信息:")
    print(f"  有效数据: {len(valid_d_wall)}/{len(df)} ({len(valid_d_wall)/len(df)*100:.1f}%)")
    if len(valid_d_wall) > 0:
        print(f"  d_wall 均值: {valid_d_wall.mean():.2f}")
        print(f"  d_wall 中位数: {valid_d_wall.median():.2f}")
        print(f"  d_wall 最小值: {valid_d_wall.min():.2f}")
        print(f"  d_wall 最大值: {valid_d_wall.max():.2f}")
        print(f"  d_wall 标准差: {valid_d_wall.std():.2f}")

    return df

def load_fsr_from_raw_csv(raw_csv_path, n_frames, frame_interval=1):

    print(f"\n从原始CSV加载FSR数据: {raw_csv_path}")

    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"找不到原始CSV文件: {raw_csv_path}")

    df_raw = pd.read_csv(raw_csv_path)

    has_raw = 'fsr_L_raw' in df_raw.columns and 'fsr_R_raw' in df_raw.columns
    has_processed = 'fsr_L' in df_raw.columns and 'fsr_R' in df_raw.columns

    if not has_raw and not has_processed:
        raise ValueError(f"CSV文件缺少FSR列 (fsr_L_raw/fsr_L): {raw_csv_path}")

    fsr_indices = list(range(0, len(df_raw), frame_interval))

    if has_raw:
        fsr_L_raw_series = df_raw['fsr_L_raw'].iloc[fsr_indices].reset_index(drop=True)
        fsr_R_raw_series = df_raw['fsr_R_raw'].iloc[fsr_indices].reset_index(drop=True)
        print(f"✓ 使用原始FSR数据 (fsr_L_raw, fsr_R_raw)")
    elif has_processed:
        fsr_L_raw_series = df_raw['fsr_L'].iloc[fsr_indices].reset_index(drop=True)
        fsr_R_raw_series = df_raw['fsr_R'].iloc[fsr_indices].reset_index(drop=True)
        print(f"✓ 使用处理后的FSR数据 (fsr_L, fsr_R)")

    original_len = len(fsr_L_raw_series)
    print(f"✓ 原始FSR采样点数: {original_len}, tracking帧数: {n_frames}")

    def _resample_fsr_series(series, target_len):

        if target_len == 0:
            return pd.Series([], dtype=float)
        if len(series) == 0:

            return pd.Series([0] * target_len, dtype=int)

        n_original = len(series)
        n_total = target_len

        original_indices = np.linspace(0, n_total - 1, n_original)
        target_indices = np.arange(n_total)

        interp_func = interp1d(
            original_indices,
            series.values.astype(float),
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )

        new_values = interp_func(target_indices)

        new_values = np.round(new_values).astype(int)

        return pd.Series(new_values)

    if original_len == n_frames:
        print("✓ FSR长度一致：直接写入（不插值）")
    elif original_len < n_frames:
        print("✓ FSR长度不足：采用“旧两步法等价”处理（尾部补齐 + 重分布插值）")

        def _pad_to_len(series, target_len):
            last_val = series.iloc[-1] if len(series) > 0 else 0
            pad = pd.Series([last_val] * (target_len - len(series)))
            return pd.concat([series, pad], ignore_index=True)

        padded_L = _pad_to_len(fsr_L_raw_series, n_frames)
        padded_R = _pad_to_len(fsr_R_raw_series, n_frames)

        tmp = pd.DataFrame({'fsr_L_raw': padded_L, 'fsr_R_raw': padded_R})
        tmp = redistribute_fsr_with_interpolation(tmp)
        fsr_L_raw_series = tmp['fsr_L_raw']
        fsr_R_raw_series = tmp['fsr_R_raw']
    else:
        print("✓ FSR长度过长：按新需求不截断，插值压缩重采样到 tracking 帧数")
        fsr_L_raw_series = _resample_fsr_series(fsr_L_raw_series, n_frames)
        fsr_R_raw_series = _resample_fsr_series(fsr_R_raw_series, n_frames)

    print(f"✓ FSR数据长度: L={len(fsr_L_raw_series)}, R={len(fsr_R_raw_series)}, 目标={n_frames}")

    return fsr_L_raw_series.values, fsr_R_raw_series.values

def _parse_calibration_dict(data):

    if "left" in data and "right" in data:
        return {
            "baseline_L": float(data["left"]["baseline"]),
            "override_L": float(data["left"]["override"]),
            "baseline_R": float(data["right"]["baseline"]),
            "override_R": float(data["right"]["override"]),
        }
    return {
        "baseline_L": float(data["baseline_L"]),
        "override_L": float(data["override_L"]),
        "baseline_R": float(data["baseline_R"]),
        "override_R": float(data["override_R"]),
    }

def resolve_fsr_calibration(
    dataset_name,
    calibration_path=None,
    baseline_L=None,
    override_L=None,
    baseline_R=None,
    override_R=None,
):

    explicit = [baseline_L, override_L, baseline_R, override_R]
    if any(v is not None for v in explicit):
        if not all(v is not None for v in explicit):
            raise ValueError(
                "When passing calibration on the command line, provide all of "
                "--baseline_L --override_L --baseline_R --override_R."
            )
        cal = {
            "baseline_L": float(baseline_L),
            "override_L": float(override_L),
            "baseline_R": float(baseline_R),
            "override_R": float(override_R),
        }
        source = "command line"
    else:
        candidates = []
        if calibration_path is not None:
            candidates.append(Path(calibration_path))
        candidates.extend([
            Path(MASK_ALIGN_DIR) / dataset_name / "fsr_calibration.json",
            Path(RAW_DATA_ROOT) / dataset_name / "fsr_calibration.json",
        ])
        chosen = next((q for q in candidates if q.exists()), None)
        if chosen is None:
            example = (
                '{"left":{"baseline":1.0,"override":4.0},'
                '"right":{"baseline":1.0,"override":4.0}}'
            )
            raise FileNotFoundError(
                "Paper-aligned FSR preprocessing requires the per-session "
                "natural-grip baseline and deliberate-override force for both "
                "hands (Eq. 4). Provide --calibration fsr_calibration.json or "
                "all four --baseline/--override values. Example JSON: " + example
            )
        with open(chosen, "r", encoding="utf-8") as f:
            cal = _parse_calibration_dict(json.load(f))
        source = str(chosen)

    for arm in ("L", "R"):
        b = cal[f"baseline_{arm}"]
        o = cal[f"override_{arm}"]
        if not np.isfinite(b) or not np.isfinite(o):
            raise ValueError(f"{arm}-arm calibration values must be finite")
        if o <= b:
            raise ValueError(
                f"{arm}-arm override force ({o}) must be greater than baseline force ({b})"
            )
    print(f"✓ FSR calibration source: {source}")
    print(
        "  L baseline/override="
        f"{cal['baseline_L']:.6g}/{cal['override_L']:.6g}; "
        "R baseline/override="
        f"{cal['baseline_R']:.6g}/{cal['override_R']:.6g}"
    )
    return cal

def normalize_fsr(df, calibration, smooth_window=3):

    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1")

    print("\n开始按论文 Eq. (4)-(5) 处理 FSR 数据...")
    for arm in ("L", "R"):
        raw_col = f"fsr_{arm}_raw"
        intent_col = f"fsr_{arm}_intent"
        out_col = f"fsr_{arm}_normalized"
        baseline = calibration[f"baseline_{arm}"]
        override = calibration[f"override_{arm}"]
        denom = override - baseline

        raw = pd.to_numeric(df[raw_col], errors="coerce")
        intent = ((raw - baseline) / denom).clip(lower=0.0, upper=1.0)
        smoothed = intent.rolling(window=smooth_window, min_periods=1).mean()
        df[intent_col] = intent
        df[out_col] = smoothed
        valid = smoothed.dropna()
        if len(valid):
            print(
                f"{arm}臂 intent: range=[{valid.min():.6f}, {valid.max():.6f}], "
                f"w={smooth_window}"
            )

    print("✅ 已完成个体化 FSR 标定与滑动平均")
    return df


def _coerce_finite_numeric_columns(df, columns):

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError("missing required numeric columns: " + ", ".join(missing))
    if len(df) == 0:
        raise ValueError("the tracking table is empty")

    converted = {}
    problems = []
    for column in columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        values = numeric.to_numpy(dtype=np.float64)
        invalid = ~np.isfinite(values)
        if invalid.any():
            positions = np.flatnonzero(invalid)
            row_examples = [str(df.index[pos]) for pos in positions[:5]]
            problems.append(
                f"{column} contains {len(positions)} non-finite/non-numeric "
                f"value(s) at row(s) {', '.join(row_examples)}"
            )
        converted[column] = numeric

    if problems:
        raise ValueError("; ".join(problems))
    return converted


def _as_finite_float32_vector(values, name):

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")

    invalid = ~np.isfinite(array)
    if invalid.any():
        positions = np.flatnonzero(invalid)
        examples = ", ".join(str(int(pos)) for pos in positions[:5])
        raise ValueError(
            f"{name} contains {len(positions)} value(s) that are not finite "
            f"float32 numbers at indices {examples}"
        )
    return array


def _prepare_alpha_labels(df, recording_dir, calibration, smooth_window):

    required = ("iou", "d_wall", "fsr_L_normalized", "fsr_R_normalized")
    numeric = _coerce_finite_numeric_columns(df, required)

    out = Path(recording_dir) / "alpha_labels.pkl"
    labels = {}
    if out.exists():
        with open(out, "rb") as f:
            labels = pickle.load(f)
        if not isinstance(labels, dict):
            raise ValueError(f"{out} must contain a dictionary")

    labels["fsr_L"] = _as_finite_float32_vector(
        numeric["fsr_L_normalized"], "fsr_L_normalized"
    )
    labels["fsr_R"] = _as_finite_float32_vector(
        numeric["fsr_R_normalized"], "fsr_R_normalized"
    )
    labels["iou"] = _as_finite_float32_vector(numeric["iou"], "iou")
    labels["d_wall"] = _as_finite_float32_vector(numeric["d_wall"], "d_wall")
    labels["fsr_calibration"] = {
        **{k: float(v) for k, v in calibration.items()},
        "smooth_window": int(smooth_window),
        "equations": "paper Eq. (4)-(5)",
    }

    for key in TRAINING_LABEL_KEYS:
        if key in labels:
            labels[key] = _as_finite_float32_vector(
                labels[key], f"alpha_labels.pkl[{key!r}]"
            )
    return out, labels


def _write_alpha_labels(out, labels):

    action = "更新已有" if out.exists() else "创建"
    print(f"✓ {action}训练标签: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(labels, f)
    print(
        "✅ alpha_labels.pkl 已写入 fsr_L/fsr_R/iou/d_wall; "
        "alpha_L/alpha_R 可在前后任一顺序由 merge_alpha_annotations.py 写入"
    )
    return out

def update_alpha_labels(df, recording_dir, calibration, smooth_window=3):

    out, labels = _prepare_alpha_labels(
        df, recording_dir, calibration, smooth_window
    )
    return _write_alpha_labels(out, labels)

def validate_data(df):

    print("\n" + "=" * 60)
    print("数据验证")
    print("=" * 60)

    try:
        numeric = _coerce_finite_numeric_columns(df, PIPELINE_NUMERIC_COLUMNS)
        for column, values in numeric.items():
            df[column] = values

        for column in ("iou", "d_wall", "fsr_L_normalized", "fsr_R_normalized"):
            _as_finite_float32_vector(df[column], column)
    except ValueError as exc:
        print(f"❌ {exc}")
        print("=" * 60)
        print("❌ 数据验证发现问题")
        print("=" * 60)
        raise

    problems = []
    for arm in ("L", "R"):
        col = f"fsr_{arm}_normalized"
        values = df[col]
        print(f"{col} 有效数据: {len(values)}/{len(df)}")
        if values.min() < -1e-8 or values.max() > 1.0 + 1e-8:
            print(f"❌ {col} 超出 [0,1]")
            problems.append(f"{col} is outside [0, 1]")
        else:
            print(f"✅ {col} range=[{values.min():.6f}, {values.max():.6f}]")

    for col in ("iou", "d_wall"):
        print(f"{col} 有效数据: {len(df)}/{len(df)}")

    if problems:
        print("=" * 60)
        print("❌ 数据验证发现问题")
        print("=" * 60)
        raise ValueError("; ".join(problems))

    print("=" * 60)
    print("✅ 数据验证完成")
    print("=" * 60)
    return True

def main(
    dataset_name=None,
    *,
    calibration_path=None,
    baseline_L=None,
    override_L=None,
    baseline_R=None,
    override_R=None,
    smooth_window=3,
    write_labels=True,
):

    if dataset_name is None:
        dataset_name = DEFAULT_DATASET_NAME

    print(f"\n==== 处理数据集: {dataset_name} ====")
    recording_dir = Path(MASK_ALIGN_DIR) / dataset_name
    csv_path = recording_dir / "tracking_results.csv"
    print(f"加载 tracking_results.csv: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ 加载了 {len(df)} 行数据")

    df = add_d_wall_column(df, dataset_name)

    raw_csv_path = Path(RAW_DATA_ROOT) / dataset_name / f"{dataset_name}.csv"
    fsr_L_raw, fsr_R_raw = load_fsr_from_raw_csv(raw_csv_path, len(df), frame_interval=1)
    df["fsr_L_raw"] = fsr_L_raw
    df["fsr_R_raw"] = fsr_R_raw
    print("✅ 已添加 FSR 原始数据列")

    calibration = resolve_fsr_calibration(
        dataset_name,
        calibration_path=calibration_path,
        baseline_L=baseline_L,
        override_L=override_L,
        baseline_R=baseline_R,
        override_R=override_R,
    )
    df = normalize_fsr(df, calibration, smooth_window=smooth_window)

    validate_data(df)
    prepared_labels = None
    if write_labels:
        prepared_labels = _prepare_alpha_labels(
            df, recording_dir, calibration, smooth_window
        )

    df.to_csv(csv_path, index=False)
    print(f"\n✅ 已保存结果到: {csv_path}")

    if prepared_labels is not None:
        _write_alpha_labels(*prepared_labels)

    print("\n✅ 处理完成！")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Merge wall-distance and paper-aligned personalized FSR data, "
            "then assemble alpha_labels.pkl"
        )
    )
    parser.add_argument("dataset", help="Dataset name under IROS_DATA_ROOT")
    parser.add_argument(
        "--calibration", type=Path, default=None,
        help="JSON containing per-session baseline/override force for both hands",
    )
    parser.add_argument("--baseline_L", type=float, default=None)
    parser.add_argument("--override_L", type=float, default=None)
    parser.add_argument("--baseline_R", type=float, default=None)
    parser.add_argument("--override_R", type=float, default=None)
    parser.add_argument(
        "--smooth_window", type=int, default=3,
        help="trailing moving-average window; paper Eq. (5) uses w=3",
    )
    parser.add_argument(
        "--no_labels", action="store_true",
        help="update tracking_results.csv only; do not write alpha_labels.pkl",
    )
    args = parser.parse_args()
    main(
        args.dataset,
        calibration_path=args.calibration,
        baseline_L=args.baseline_L,
        override_L=args.override_L,
        baseline_R=args.baseline_R,
        override_R=args.override_R,
        smooth_window=args.smooth_window,
        write_labels=not args.no_labels,
    )
