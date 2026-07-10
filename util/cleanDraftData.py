#!/usr/bin/env python3
"""Clean draft pick logs into model-ready X/y chunks.

X keeps the existing draft-state columns:
  pack_number, pick_number, every pack_card_* column, every pool_* column

y is the integer card index of the picked card, using the same card order as
the dataset's pack_card_* columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/draft_data_public.SOS.PremierDraft.csv")
DEFAULT_OUTPUT_DIR = Path("data/cleaned")
PACK_PREFIX = "pack_card_"
POOL_PREFIX = "pool_"


def read_columns(csv_path: Path) -> list[str]:
    return pd.read_csv(csv_path, nrows=0).columns.tolist()


def get_card_names(columns: list[str]) -> list[str]:
    return [column.removeprefix(PACK_PREFIX) for column in columns if column.startswith(PACK_PREFIX)]


def get_feature_columns(columns: list[str]) -> list[str]:
    pack_columns = [column for column in columns if column.startswith(PACK_PREFIX)]
    pool_columns = [column for column in columns if column.startswith(POOL_PREFIX)]
    return ["pack_number", "pick_number", *pack_columns, *pool_columns]


def validate_columns(columns: list[str], card_names: list[str], feature_columns: list[str]) -> None:
    missing = [column for column in ["pack_number", "pick_number", "pick"] if column not in columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    pool_names = [column.removeprefix(POOL_PREFIX) for column in columns if column.startswith(POOL_PREFIX)]
    if card_names != pool_names:
        raise ValueError("pack_card_* and pool_* columns do not contain the same cards in the same order.")

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Feature columns contain duplicates.")


def make_metadata(
    csv_path: Path,
    feature_columns: list[str],
    card_names: list[str],
    rows_written: int,
    chunks_written: int,
) -> dict[str, Any]:
    return {
        "source_csv": str(csv_path),
        "rows": rows_written,
        "chunks": chunks_written,
        "x_dtype": "uint8",
        "y_dtype": "int16",
        "x_columns": feature_columns,
        "y_column": "pick",
        "y_encoding": "card_index",
        "card_index_source": "Order of pack_card_* columns in the source CSV header.",
        "card_names": card_names,
        "card_to_index": {card_name: index for index, card_name in enumerate(card_names)},
        "index_to_card": {str(index): card_name for index, card_name in enumerate(card_names)},
    }


def save_chunk(
    chunk: pd.DataFrame,
    feature_columns: list[str],
    card_to_index: dict[str, int],
    output_dir: Path,
    chunk_number: int,
) -> int:
    unknown_picks = sorted(set(chunk["pick"].dropna()) - set(card_to_index))
    if unknown_picks:
        preview = ", ".join(unknown_picks[:10])
        raise ValueError(f"Found pick values that are not in the card index: {preview}")

    x = chunk[feature_columns].to_numpy(dtype=np.uint8, copy=True)
    y = chunk["pick"].map(card_to_index).to_numpy(dtype=np.int16, copy=True)

    output_path = output_dir / f"draft_xy_chunk_{chunk_number:04d}.npz"
    np.savez_compressed(output_path, X=x, y=y)
    return len(chunk)


def clean_draft_data(
    csv_path: Path,
    output_dir: Path,
    chunksize: int,
    limit_rows: int | None,
) -> None:
    columns = read_columns(csv_path)
    card_names = get_card_names(columns)
    feature_columns = get_feature_columns(columns)
    validate_columns(columns, card_names, feature_columns)

    card_to_index = {card_name: index for index, card_name in enumerate(card_names)}
    usecols = [*feature_columns, "pick"]
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    chunks_written = 0

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        if limit_rows is not None:
            remaining = limit_rows - rows_written
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        rows_written += save_chunk(
            chunk=chunk,
            feature_columns=feature_columns,
            card_to_index=card_to_index,
            output_dir=output_dir,
            chunk_number=chunks_written,
        )
        chunks_written += 1
        print(f"Wrote chunk {chunks_written} ({rows_written:,} total rows)")

    metadata = make_metadata(
        csv_path=csv_path,
        feature_columns=feature_columns,
        card_names=card_names,
        rows_written=rows_written,
        chunks_written=chunks_written,
    )
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata to {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create X/y chunks from a 17Lands draft CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source draft CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output folder.")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Rows per saved .npz chunk.")
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional row limit for quick tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clean_draft_data(
        csv_path=args.input,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        limit_rows=args.limit_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
