#!/usr/bin/env python3
"""Add deckbuilder targets to cleaned draft .npz chunks.

The cleaned draft data is pick-level:
  X = [pack_number, pick_number, pack_counts..., pool_counts...]
  y = picked card index

Only final-pick rows have a completed draft pool. For those rows, this script
adds deckbuilder targets using the exact metadata card order that the neural
network uses. Non-final rows get zero deck arrays and has_deck=False.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SL.util.deckbuilding import (  # noqa: E402
    BASIC_LANDS,
    COLORLESS_LAND,
    build_deck,
    load_ratings_table,
)


DEFAULT_INPUT_DIR = ROOT / "SL" / "data" / "cleaned"
DEFAULT_OUTPUT_DIR = ROOT / "SL" / "data" / "cleaned_with_decks"
DEFAULT_RATINGS = ROOT / "card-ratings-2026-07-14 (1).csv"
DEFAULT_CARDS = ROOT / "SOS_cards.json"
DEFAULT_COMPLETE_POOL_SIZE = 42
PACK_PREFIX = "pack_card_"
POOL_PREFIX = "pool_"
DECK_COUNTS_KEY = "deck_counts"
DECK_SPELL_COUNTS_KEY = "deck_spell_counts"
HAS_DECK_KEY = "has_deck"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def validate_metadata(metadata: dict[str, Any]) -> list[str]:
    card_names = list(metadata["card_names"])
    card_to_index = metadata.get("card_to_index")
    if card_to_index is not None:
        expected = {card_name: index for index, card_name in enumerate(card_names)}
        if card_to_index != expected:
            raise ValueError("metadata card_to_index does not match card_names order.")

    x_columns = metadata.get("x_columns", [])
    expected_width = 2 + 2 * len(card_names)
    if len(x_columns) != expected_width:
        raise ValueError(f"Expected {expected_width} X columns, found {len(x_columns)}.")

    pack_columns = [f"{PACK_PREFIX}{card_name}" for card_name in card_names]
    pool_columns = [f"{POOL_PREFIX}{card_name}" for card_name in card_names]
    if x_columns[2 : 2 + len(card_names)] != pack_columns:
        raise ValueError("pack_card_* columns do not match metadata card_names order.")
    if x_columns[2 + len(card_names) : 2 + 2 * len(card_names)] != pool_columns:
        raise ValueError("pool_* columns do not match metadata card_names order.")

    return card_names


def add_pick_to_pool(pool_counts: np.ndarray, picked_index: int) -> np.ndarray:
    pool_after_pick = pool_counts.astype(np.uint8, copy=True)
    pool_after_pick[picked_index] += 1
    return pool_after_pick


def names_to_counts(card_names: list[str], card_to_index: dict[str, int], deck: list[str]) -> np.ndarray:
    counts = np.zeros(len(card_names), dtype=np.uint8)
    unknown = sorted(set(deck) - set(card_to_index))
    if unknown:
        preview = ", ".join(unknown[:10])
        raise ValueError(f"Deckbuilder returned card(s) outside metadata card index: {preview}")

    for card_name in deck:
        counts[card_to_index[card_name]] += 1
    return counts


def validate_deck_from_pool(
    deck_counts: np.ndarray,
    spell_counts: np.ndarray,
    pool_counts: np.ndarray,
    card_names: list[str],
    deck: list[str],
    row_label: str,
) -> None:
    generated_land_names = set(BASIC_LANDS.values()) | {COLORLESS_LAND}
    generated_land_counts = np.zeros_like(deck_counts)
    deck_counter = Counter(deck)

    for land_name in generated_land_names:
        if land_name in card_names:
            generated_land_counts[card_names.index(land_name)] = deck_counter[land_name]

    if np.any(spell_counts > pool_counts):
        bad_indices = np.flatnonzero(spell_counts > pool_counts)
        bad = ", ".join(
            f"{card_names[i]} deck={int(spell_counts[i])} pool={int(pool_counts[i])}"
            for i in bad_indices[:10]
        )
        raise ValueError(f"{row_label}: deckbuilder selected spell card(s) not in pool: {bad}")

    required_from_pool = deck_counts - generated_land_counts
    if np.any(required_from_pool > pool_counts):
        bad_indices = np.flatnonzero(required_from_pool > pool_counts)
        bad = ", ".join(
            f"{card_names[i]} deck={int(required_from_pool[i])} pool={int(pool_counts[i])}"
            for i in bad_indices[:10]
        )
        raise ValueError(f"{row_label}: full deck contains non-generated card(s) not in pool: {bad}")


def build_targets_for_row(
    pool_after_pick: np.ndarray,
    card_names: list[str],
    card_to_index: dict[str, int],
    index_to_card: dict[int, str],
    table,
    total_lands: int,
    expected_deck_size: int,
    expected_spell_count: int,
    row_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    spell_deck = build_deck(
        pool_after_pick,
        index_to_card,
        table,
        add_lands=False,
    )
    full_deck = build_deck(
        pool_after_pick,
        index_to_card,
        table,
        add_lands=True,
        total_lands=total_lands,
    )

    if len(spell_deck) != expected_spell_count:
        raise ValueError(f"{row_label}: expected {expected_spell_count} spells, got {len(spell_deck)}.")
    if len(full_deck) != expected_deck_size:
        raise ValueError(f"{row_label}: expected {expected_deck_size} deck cards, got {len(full_deck)}.")

    spell_counts = names_to_counts(card_names, card_to_index, spell_deck)
    deck_counts = names_to_counts(card_names, card_to_index, full_deck)
    validate_deck_from_pool(deck_counts, spell_counts, pool_after_pick, card_names, full_deck, row_label)
    return deck_counts, spell_counts


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp_path, **arrays)
    tmp_npz_path = tmp_path.with_suffix(tmp_path.suffix + ".npz")
    if tmp_npz_path.exists():
        tmp_npz_path.replace(path)
    else:
        tmp_path.replace(path)


def process_chunk(
    input_path: Path,
    output_path: Path,
    card_names: list[str],
    table,
    complete_pool_size: int,
    total_lands: int,
    expected_spell_count: int,
    skip_invalid: bool,
) -> tuple[int, int, int]:
    card_count = len(card_names)
    card_to_index = {card_name: index for index, card_name in enumerate(card_names)}
    index_to_card = {index: card_name for index, card_name in enumerate(card_names)}

    with np.load(input_path) as data:
        arrays = {key: data[key] for key in data.files}

    if "X" not in arrays or "y" not in arrays:
        raise ValueError(f"{input_path} must contain X and y arrays.")
    X = arrays["X"]
    y = arrays["y"]
    expected_width = 2 + 2 * card_count
    if X.ndim != 2 or X.shape[1] != expected_width:
        raise ValueError(f"{input_path} X shape {X.shape} does not match expected width {expected_width}.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"{input_path} y shape {y.shape} does not match X rows {X.shape[0]}.")

    deck_counts = np.zeros((X.shape[0], card_count), dtype=np.uint8)
    deck_spell_counts = np.zeros((X.shape[0], card_count), dtype=np.uint8)
    has_deck = np.zeros(X.shape[0], dtype=np.bool_)

    pool_start = 2 + card_count
    pool_stop = pool_start + card_count
    rows_seen = rows_with_decks = invalid_rows = 0

    for row_index in range(X.shape[0]):
        pool_before_pick = X[row_index, pool_start:pool_stop]
        pool_after_pick = add_pick_to_pool(pool_before_pick, int(y[row_index]))
        if int(pool_after_pick.sum()) < complete_pool_size:
            continue

        rows_seen += 1
        row_label = f"{input_path.name} row {row_index}"
        try:
            row_deck_counts, row_spell_counts = build_targets_for_row(
                pool_after_pick=pool_after_pick,
                card_names=card_names,
                card_to_index=card_to_index,
                index_to_card=index_to_card,
                table=table,
                total_lands=total_lands,
                expected_deck_size=expected_spell_count + total_lands,
                expected_spell_count=expected_spell_count,
                row_label=row_label,
            )
        except ValueError:
            invalid_rows += 1
            if skip_invalid:
                continue
            raise

        deck_counts[row_index] = row_deck_counts
        deck_spell_counts[row_index] = row_spell_counts
        has_deck[row_index] = True
        rows_with_decks += 1

    arrays[DECK_COUNTS_KEY] = deck_counts
    arrays[DECK_SPELL_COUNTS_KEY] = deck_spell_counts
    arrays[HAS_DECK_KEY] = has_deck

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_npz(output_path, arrays)
    return rows_seen, rows_with_decks, invalid_rows


def update_metadata(
    metadata: dict[str, Any],
    output_dir: Path,
    input_dir: Path,
    ratings_path: Path,
    cards_path: Path,
    complete_pool_size: int,
    total_lands: int,
    expected_spell_count: int,
    chunks_written: int,
    rows_with_complete_pool: int,
    rows_with_decks: int,
    invalid_rows: int,
) -> None:
    metadata = dict(metadata)
    metadata["deckbuilder_targets"] = {
        "source": "RL.deckbuilding.build_deck",
        "input_dir": str(input_dir),
        "ratings_path": str(ratings_path),
        "cards_path": str(cards_path),
        "card_index_source": "Same metadata card_names order used by X/y and the neural network.",
        "deck_counts_key": DECK_COUNTS_KEY,
        "deck_spell_counts_key": DECK_SPELL_COUNTS_KEY,
        "has_deck_key": HAS_DECK_KEY,
        "complete_pool_size": complete_pool_size,
        "expected_spell_count": expected_spell_count,
        "total_lands": total_lands,
        "expected_deck_size": expected_spell_count + total_lands,
        "generated_basic_lands": BASIC_LANDS,
        "colorless_land": COLORLESS_LAND,
        "chunks_written": chunks_written,
        "rows_with_complete_pool": rows_with_complete_pool,
        "rows_with_decks": rows_with_decks,
        "invalid_rows": invalid_rows,
    }
    metadata["phase_aware_training"] = {
        "model_class": "SL.NN.DraftScorer",
        "num_cards": len(metadata["card_names"]),
        "forward_inputs": {
            "pool_counts": "Draft phase: X pool columns. Build phase: completed pool after adding y to the final-pick row.",
            "pack_counts": "Draft phase: X pack columns. Build phase: zeros.",
            "deck_counts": "Draft phase: zeros. Build phase: current deck being built up from the completed pool.",
            "phases": "0 for draft-pick rows, 1 for deckbuilding add rows.",
            "pack_numbers": "X pack_number.",
            "pick_numbers": "X pick_number.",
            "build_steps": "0 for draft-pick rows; sequential add step number for deckbuilding rows.",
        },
        "draft_phase": {
            "target": "y",
            "legal_mask": "pack_counts > 0",
            "action_meaning": "pick this card from the current pack",
        },
        "build_phase": {
            "source_rows": "Rows where has_deck is true.",
            "initial_pool": "X pool columns plus one count at y.",
            "initial_deck_counts": "zeros",
            "target_deck_counts": DECK_COUNTS_KEY,
            "spell_target_deck_counts": DECK_SPELL_COUNTS_KEY,
            "target_mask": "current deck_counts < target_deck_counts",
            "legal_mask": "pool_counts > deck_counts OR action is a generated basic land.",
            "action_meaning": "add this drafted card from the completed pool or append a basic land to the current deck",
            "stop_condition": "deck_counts == target_deck_counts",
        },
    }
    save_json(output_dir / "metadata.json", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add deckbuilder targets to cleaned draft .npz chunks.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument("--complete-pool-size", type=int, default=DEFAULT_COMPLETE_POOL_SIZE)
    parser.add_argument("--total-lands", type=int, default=17)
    parser.add_argument("--spell-count", type=int, default=23)
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping rows deckbuilder cannot complete.")
    parser.add_argument("--in-place", action="store_true", help="Write back into input-dir instead of output-dir.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = input_dir if args.in_place else args.output_dir.resolve()
    metadata_path = input_dir / "metadata.json"
    metadata = load_json(metadata_path)
    card_names = validate_metadata(metadata)

    missing_basics = [name for name in BASIC_LANDS.values() if name not in card_names]
    if missing_basics:
        raise ValueError(f"Missing basic land(s) from neural card index: {', '.join(missing_basics)}")

    table = load_ratings_table(args.ratings, args.cards)
    chunks = sorted(input_dir.glob("draft_xy_chunk_*.npz"))
    if args.limit_chunks is not None:
        chunks = chunks[: args.limit_chunks]
    if not chunks:
        raise ValueError(f"No draft_xy_chunk_*.npz files found in {input_dir}")

    total_complete = total_decks = total_invalid = 0
    for chunk_number, input_path in enumerate(chunks, 1):
        output_path = output_dir / input_path.name
        complete, decks, invalid = process_chunk(
            input_path=input_path,
            output_path=output_path,
            card_names=card_names,
            table=table,
            complete_pool_size=args.complete_pool_size,
            total_lands=args.total_lands,
            expected_spell_count=args.spell_count,
            skip_invalid=not args.strict,
        )
        total_complete += complete
        total_decks += decks
        total_invalid += invalid
        print(
            f"{chunk_number:>3}/{len(chunks)} {input_path.name}: "
            f"complete pools {complete:,}, decks {decks:,}, invalid {invalid:,}"
        )

    update_metadata(
        metadata=metadata,
        output_dir=output_dir,
        input_dir=input_dir,
        ratings_path=args.ratings,
        cards_path=args.cards,
        complete_pool_size=args.complete_pool_size,
        total_lands=args.total_lands,
        expected_spell_count=args.spell_count,
        chunks_written=len(chunks),
        rows_with_complete_pool=total_complete,
        rows_with_decks=total_decks,
        invalid_rows=total_invalid,
    )
    print(f"Wrote deckbuilder targets to {output_dir}")
    print(f"Rows with complete pools: {total_complete:,}; rows with decks: {total_decks:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
