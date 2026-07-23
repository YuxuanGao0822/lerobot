#!/usr/bin/env python

"""Build a local RoboTwin dataset overlay with exact state/action quantiles.

The public ``lerobot/robotwin_unified`` snapshot predates LeRobot's quantile
statistics. Pi0 and Pi0.5 require q01/q99 normalization for state and action.
The generic augmentation script decodes every video frame and pushes its
result to the Hub, which is unnecessary for post-training.

This remote-only preparation utility instead:

* scans only ``observation.state`` and ``action`` in the Parquet files;
* computes exact q01/q10/q50/q90/q99 values over all dataset frames;
* copies the small ``meta/`` tree into a derived local dataset root;
* symlinks ``data/`` and ``videos/`` so the 79.5 GB snapshot is not duplicated;
* never modifies the source Hub snapshot and never uploads anything.

It is intentionally kept outside the training package. Run it once on the
remote server, then set ``ROBOTWIN_DATASET_ROOT`` when invoking run_one.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads


REVISION = "1287871839fae2296bc27b88a5457c3e1eba8e1f"
REPO_CACHE_NAME = "datasets--lerobot--robotwin_unified"
TARGET_FEATURES = ("observation.state", "action")
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def default_lerobot_home() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return Path(os.environ.get("HF_LEROBOT_HOME", hf_home / "lerobot")).expanduser()


def default_source_root() -> Path:
    return default_lerobot_home() / "hub" / REPO_CACHE_NAME / "snapshots" / REVISION


def default_output_root() -> Path:
    return default_lerobot_home() / "derived" / f"robotwin_unified_quantiles_{REVISION[:12]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a no-copy RoboTwin dataset overlay with exact state/action quantiles."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=default_source_root(),
        help="Resolved source snapshot containing meta/, data/, and videos/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help="New local dataset root. It must not already contain unrelated files.",
    )
    return parser.parse_args()


def validate_source(source_root: Path) -> list[Path]:
    required = (source_root / "meta" / "info.json", source_root / "meta" / "stats.json")
    missing = [path for path in required if not path.is_file()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Source snapshot is incomplete; missing:\n{rendered}")

    parquet_files = sorted((source_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found below {source_root / 'data'}")
    return parquet_files


def column_to_matrix(column: pa.ChunkedArray, feature: str) -> np.ndarray:
    array = column.combine_chunks()

    if pa.types.is_fixed_size_list(array.type):
        matrix = array.values.to_numpy(zero_copy_only=False).reshape(len(array), array.type.list_size)
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = array.offsets.to_numpy(zero_copy_only=False)
        widths = np.diff(offsets)
        if len(widths) == 0:
            raise ValueError(f"Feature {feature!r} has no rows")
        if not np.all(widths == widths[0]):
            raise ValueError(f"Feature {feature!r} has variable-length rows and cannot be normalized")
        start = int(offsets[0])
        stop = int(offsets[-1])
        flat = array.values.slice(start, stop - start).to_numpy(zero_copy_only=False)
        matrix = flat.reshape(len(array), int(widths[0]))
    elif pa.types.is_floating(array.type) or pa.types.is_integer(array.type):
        matrix = array.to_numpy(zero_copy_only=False).reshape(-1, 1)
    else:
        # This fallback is deliberately explicit: it supports unusual Arrow
        # encodings while keeping the common fixed-size-list path fast.
        matrix = np.asarray(array.to_pylist())

    if matrix.ndim != 2:
        raise ValueError(f"Expected a rank-2 matrix for {feature!r}, got shape {matrix.shape}")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"Feature {feature!r} is not numeric: dtype={matrix.dtype}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Feature {feature!r} contains NaN or infinity")
    return matrix


def compute_quantiles(parquet_files: list[Path], feature: str) -> tuple[int, dict[str, list[float]]]:
    dataset = pads.dataset([str(path) for path in parquet_files], format="parquet")
    if feature not in dataset.schema.names:
        raise KeyError(f"Feature {feature!r} is absent from Parquet schema: {dataset.schema.names}")

    # Read one feature at a time. For 6.1M RoboTwin frames and 14 dimensions,
    # this stays well below the memory required to materialize the full table.
    table = dataset.to_table(columns=[feature], use_threads=True)
    matrix = column_to_matrix(table.column(feature), feature)
    values = np.quantile(matrix, QUANTILES, axis=0)
    quantile_stats = {
        f"q{int(quantile * 100):02d}": values[index].astype(np.float64).tolist()
        for index, quantile in enumerate(QUANTILES)
    }
    return matrix.shape[0], quantile_stats


def make_overlay(source_root: Path, output_root: Path, parquet_files: list[Path]) -> None:
    source_root = source_root.resolve()
    output_root = output_root.expanduser().absolute()
    if source_root == output_root:
        raise ValueError("--output-root must differ from --source-root")

    marker = output_root / "robotwin_quantile_overlay.json"
    if output_root.exists():
        if marker.is_file():
            print(f"Overlay already exists: {output_root}")
            print(f"Use: ROBOTWIN_DATASET_ROOT={output_root}")
            return
        raise FileExistsError(
            f"Refusing to overwrite existing non-overlay directory: {output_root}\n"
            "Choose another --output-root or archive the directory first."
        )

    temporary_root = output_root.parent / f".{output_root.name}.preparing-{os.getpid()}"
    if temporary_root.exists():
        raise FileExistsError(f"Temporary preparation directory already exists: {temporary_root}")
    temporary_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root.mkdir()

    try:
        shutil.copytree(source_root / "meta", temporary_root / "meta")
        for directory_name in ("data", "videos"):
            source_directory = source_root / directory_name
            if source_directory.exists():
                (temporary_root / directory_name).symlink_to(source_directory, target_is_directory=True)

        stats_path = temporary_root / "meta" / "stats.json"
        with stats_path.open(encoding="utf-8") as stats_file:
            stats = json.load(stats_file)

        row_counts: dict[str, int] = {}
        for feature in TARGET_FEATURES:
            print(f"Computing exact quantiles for {feature!r} from {len(parquet_files)} Parquet files...")
            row_count, feature_quantiles = compute_quantiles(parquet_files, feature)
            if feature not in stats:
                raise KeyError(f"Feature {feature!r} is absent from meta/stats.json")
            stats[feature].update(feature_quantiles)
            row_counts[feature] = row_count
            print(
                f"  rows={row_count}; dim={len(feature_quantiles['q01'])}; "
                f"q01/q99 added"
            )

        if len(set(row_counts.values())) != 1:
            raise ValueError(f"State/action row counts disagree: {row_counts}")

        temporary_stats_path = stats_path.with_suffix(".json.tmp")
        with temporary_stats_path.open("w", encoding="utf-8") as stats_file:
            json.dump(stats, stats_file, indent=2, allow_nan=False)
            stats_file.write("\n")
        temporary_stats_path.replace(stats_path)

        marker_payload = {
            "source_root": str(source_root),
            "source_revision": REVISION,
            "features": list(TARGET_FEATURES),
            "quantiles": list(QUANTILES),
            "row_counts": row_counts,
            "data_and_videos_are_symlinked": True,
        }
        with (temporary_root / marker.name).open("w", encoding="utf-8") as marker_file:
            json.dump(marker_payload, marker_file, indent=2)
            marker_file.write("\n")

        temporary_root.replace(output_root)
    except Exception:
        print(f"Preparation failed; partial files were left for inspection at: {temporary_root}")
        raise

    print(f"Prepared RoboTwin quantile overlay: {output_root}")
    print(f"Use: ROBOTWIN_DATASET_ROOT={output_root}")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser()
    parquet_files = validate_source(source_root)
    make_overlay(source_root, args.output_root, parquet_files)


if __name__ == "__main__":
    main()
