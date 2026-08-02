#!/usr/bin/env python3
"""Replace generated deckbuilder targets with actual 17Lands game decks.

The cleaned draft chunks do not store draft_id, so this script replays the
same draft CSV filtering/chunking used by cleanDraftData.py to recover the
draft_id for each row. It then writes actual main-deck targets from game_data
onto the terminal draft rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "SL" / "data" / "cleaned_with_decks"
DEFAULT_DRAFT_CSV = ROOT / "data" / "raw" / "draft_data_public.SOS.PremierDraft.csv"
DEFAULT_GAME_CSV = ROOT / "data" / "raw" / "game_data_public.SOS.PremierDraft.csv"
PACK_PREFIX = "pack_card_"
POOL_PREFIX = "pool_"
DECK_PREFIX = "deck_"
DECK_COUNTS_KEY = "deck_counts"
DECK_SPELL_COUNTS_KEY = "deck_spell_counts"
HAS_DECK_KEY = "has_deck"
BASIC_LAND_NAMES = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}


@dataclass(frozen=True)
class TerminalRow:
    chunk_index: int
    row_index: int
    draft_id: str
    final_pool: np.ndarray


@dataclass(frozen=True)
class SelectedDeck:
    sort_key: tuple[Any, ...]
    counts: np.ndarray
    build_index: int
    match_number: int
    game_number: int
    game_time: str


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp_path, **arrays)
    tmp_npz_path = tmp_path.with_suffix(tmp_path.suffix + ".npz")
    if tmp_npz_path.exists():
        tmp_npz_path.replace(path)
    else:
        tmp_path.replace(path)


def draft_chunk_paths(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.glob("draft_xy_chunk_*.npz")
        if path.stem.removeprefix("draft_xy_chunk_").isdigit()
    )


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


def collect_terminal_rows(
    data_dir: Path,
    draft_csv: Path,
    card_count: int,
    chunksize: int,
    min_match_wins: int,
    complete_pool_size: int,
) -> list[TerminalRow]:
    chunks = draft_chunk_paths(data_dir)
    if not chunks:
        raise ValueError(f"No draft_xy_chunk_*.npz files found in {data_dir}")

    terminal_rows: list[TerminalRow] = []
    seen_draft_ids: set[str] = set()
    chunk_index = 0
    usecols = ["draft_id", "event_match_wins"]

    for csv_chunk in pd.read_csv(draft_csv, usecols=usecols, chunksize=chunksize):
        csv_chunk = csv_chunk[csv_chunk["event_match_wins"] > min_match_wins].reset_index(drop=True)
        if csv_chunk.empty:
            continue
        if chunk_index >= len(chunks):
            raise ValueError("Draft CSV produced more cleaned chunks than the data directory contains.")

        chunk_path = chunks[chunk_index]
        with np.load(chunk_path) as data:
            X = data["X"]
            y = data["y"]
            if len(csv_chunk) != y.shape[0]:
                raise ValueError(
                    f"{chunk_path.name}: cleaned row count {y.shape[0]:,} does not match "
                    f"replayed CSV row count {len(csv_chunk):,}. Check --chunksize and --min-match-wins."
                )
            pool_start = 2 + card_count
            pool_stop = pool_start + card_count
            pool_counts = X[:, pool_start:pool_stop].astype(np.uint16, copy=False)
            final_pool_sizes = pool_counts.sum(axis=1) + 1
            final_rows = np.flatnonzero(final_pool_sizes >= complete_pool_size)

            for row_index in final_rows:
                row_index = int(row_index)
                draft_id = str(csv_chunk.at[row_index, "draft_id"])
                if draft_id in seen_draft_ids:
                    raise ValueError(f"Duplicate terminal draft_id in cleaned data: {draft_id}")
                seen_draft_ids.add(draft_id)

                final_pool = pool_counts[row_index].astype(np.uint16, copy=True)
                final_pool[int(y[row_index])] += 1
                terminal_rows.append(
                    TerminalRow(
                        chunk_index=chunk_index,
                        row_index=row_index,
                        draft_id=draft_id,
                        final_pool=final_pool,
                    )
                )

        chunk_index += 1

    if chunk_index != len(chunks):
        raise ValueError(
            f"Draft CSV replay matched {chunk_index:,} chunks, but {len(chunks):,} chunk files exist."
        )
    return terminal_rows


def validate_game_columns(game_csv: Path, card_names: list[str]) -> list[str]:
    columns = pd.read_csv(game_csv, nrows=0).columns.tolist()
    deck_columns = [f"{DECK_PREFIX}{card_name}" for card_name in card_names]
    missing = [column for column in deck_columns if column not in columns]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Game CSV is missing deck column(s): {preview}")
    return deck_columns


def game_sort_key(row: tuple[Any, ...]) -> tuple[Any, ...]:
    _, game_time, build_index, match_number, game_number = row[:5]
    return (
        "" if pd.isna(game_time) else str(game_time),
        -1 if pd.isna(match_number) else int(match_number),
        -1 if pd.isna(game_number) else int(game_number),
        -1 if pd.isna(build_index) else int(build_index),
    )


def collect_latest_game_decks(
    game_csv: Path,
    card_names: list[str],
    relevant_draft_ids: set[str],
    chunksize: int,
) -> tuple[dict[str, SelectedDeck], Counter[int]]:
    deck_columns = validate_game_columns(game_csv, card_names)
    usecols = ["draft_id", "game_time", "build_index", "match_number", "game_number", *deck_columns]
    selected: dict[str, SelectedDeck] = {}
    build_indices_by_draft: dict[str, set[int]] = defaultdict(set)
    card_count = len(card_names)

    for chunk in pd.read_csv(game_csv, usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["draft_id"].isin(relevant_draft_ids)]
        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False, name=None):
            draft_id = str(row[0])
            key = game_sort_key(row)
            build_index = key[3]
            build_indices_by_draft[draft_id].add(build_index)
            if draft_id in selected and key <= selected[draft_id].sort_key:
                continue

            counts = np.fromiter((int(value) for value in row[5:]), dtype=np.uint16, count=card_count)
            selected[draft_id] = SelectedDeck(
                sort_key=key,
                counts=counts,
                build_index=build_index,
                match_number=key[1],
                game_number=key[2],
                game_time=key[0],
            )

    build_count_distribution = Counter(len(indices) for indices in build_indices_by_draft.values())
    return selected, build_count_distribution


def validate_deck_against_pool(
    deck_counts: np.ndarray,
    final_pool: np.ndarray,
    basic_land_mask: np.ndarray,
    card_names: list[str],
) -> list[str]:
    invalid = np.flatnonzero((deck_counts > final_pool) & ~basic_land_mask)
    return [
        f"{card_names[index]} deck={int(deck_counts[index])} pool={int(final_pool[index])}"
        for index in invalid
    ]


def replace_chunk_targets(
    data_dir: Path,
    terminal_rows: list[TerminalRow],
    selected_decks: dict[str, SelectedDeck],
    card_names: list[str],
    strict: bool,
) -> dict[str, Any]:
    chunks = draft_chunk_paths(data_dir)
    card_count = len(card_names)
    basic_land_mask = np.asarray([card_name in BASIC_LAND_NAMES for card_name in card_names], dtype=np.bool_)
    rows_by_chunk: dict[int, list[TerminalRow]] = defaultdict(list)
    for row in terminal_rows:
        rows_by_chunk[row.chunk_index].append(row)

    total_terminal = 0
    rows_with_decks = 0
    missing_game_rows = 0
    invalid_pool_rows = 0
    invalid_examples: list[dict[str, Any]] = []
    deck_size_distribution: Counter[int] = Counter()
    drafted_card_count_distribution: Counter[int] = Counter()
    basic_land_count_distribution: Counter[int] = Counter()
    selected_build_index_distribution: Counter[int] = Counter()

    for chunk_index, chunk_path in enumerate(chunks):
        with np.load(chunk_path) as data:
            arrays = {key: data[key] for key in data.files}

        X = arrays["X"]
        deck_counts = np.zeros((X.shape[0], card_count), dtype=np.uint8)
        deck_spell_counts = np.zeros((X.shape[0], card_count), dtype=np.uint8)
        has_deck = np.zeros(X.shape[0], dtype=np.bool_)

        for terminal_row in rows_by_chunk.get(chunk_index, []):
            total_terminal += 1
            selected = selected_decks.get(terminal_row.draft_id)
            if selected is None:
                missing_game_rows += 1
                if strict:
                    raise ValueError(f"{terminal_row.draft_id}: no matching game_data deck row.")
                continue

            bad_cards = validate_deck_against_pool(
                selected.counts,
                terminal_row.final_pool,
                basic_land_mask,
                card_names,
            )
            if bad_cards:
                invalid_pool_rows += 1
                invalid_examples.append(
                    {
                        "draft_id": terminal_row.draft_id,
                        "chunk": chunk_path.name,
                        "row": terminal_row.row_index,
                        "bad_cards": bad_cards[:10],
                    }
                )
                if strict:
                    raise ValueError(f"{terminal_row.draft_id}: actual deck contains card(s) not in pool: {bad_cards[:10]}")
                continue

            if int(selected.counts.max()) > np.iinfo(np.uint8).max:
                raise ValueError(f"{terminal_row.draft_id}: deck count exceeds uint8 capacity.")

            deck_u8 = selected.counts.astype(np.uint8, copy=False)
            spell_u8 = deck_u8.copy()
            spell_u8[basic_land_mask] = 0

            deck_counts[terminal_row.row_index] = deck_u8
            deck_spell_counts[terminal_row.row_index] = spell_u8
            has_deck[terminal_row.row_index] = True
            rows_with_decks += 1

            deck_size_distribution[int(deck_u8.sum())] += 1
            drafted_card_count_distribution[int(spell_u8.sum())] += 1
            basic_land_count_distribution[int(deck_u8[basic_land_mask].sum())] += 1
            selected_build_index_distribution[selected.build_index] += 1

        arrays[DECK_COUNTS_KEY] = deck_counts
        arrays[DECK_SPELL_COUNTS_KEY] = deck_spell_counts
        arrays[HAS_DECK_KEY] = has_deck
        write_npz(chunk_path, arrays)
        print(f"{chunk_index + 1:>3}/{len(chunks)} {chunk_path.name}: decks {int(has_deck.sum()):,}")

    return {
        "terminal_rows": total_terminal,
        "rows_with_decks": rows_with_decks,
        "missing_game_rows": missing_game_rows,
        "invalid_pool_rows": invalid_pool_rows,
        "invalid_examples": invalid_examples[:20],
        "deck_size_distribution": dict(sorted(deck_size_distribution.items())),
        "drafted_card_count_distribution": dict(sorted(drafted_card_count_distribution.items())),
        "basic_land_count_distribution": dict(sorted(basic_land_count_distribution.items())),
        "selected_build_index_distribution": dict(sorted(selected_build_index_distribution.items())),
    }


def update_metadata(
    data_dir: Path,
    metadata: dict[str, Any],
    draft_csv: Path,
    game_csv: Path,
    chunksize: int,
    min_match_wins: int,
    complete_pool_size: int,
    build_count_distribution: Counter[int],
    replacement_stats: dict[str, Any],
) -> None:
    metadata = dict(metadata)
    target_metadata = {
        "source": "17Lands game_data deck_* columns",
        "selection": "latest game row per draft_id by game_time, then match_number, game_number, build_index",
        "draft_csv": str(draft_csv),
        "game_csv": str(game_csv),
        "data_dir": str(data_dir),
        "card_index_source": "Same metadata card_names order used by X/y and the neural network.",
        "deck_counts_key": DECK_COUNTS_KEY,
        "deck_spell_counts_key": DECK_SPELL_COUNTS_KEY,
        "has_deck_key": HAS_DECK_KEY,
        "deck_spell_counts_meaning": "Actual main deck with generated basic lands zeroed out.",
        "chunksize": chunksize,
        "min_match_wins": min_match_wins,
        "complete_pool_size": complete_pool_size,
        "chunks_written": metadata.get("chunks"),
        "build_count_distribution_per_draft": dict(sorted(build_count_distribution.items())),
        **replacement_stats,
    }
    metadata["deckbuilder_targets"] = target_metadata
    metadata["actual_deck_targets"] = target_metadata

    phase_training = metadata.get("phase_aware_training")
    if isinstance(phase_training, dict):
        build_phase = phase_training.get("build_phase")
        if isinstance(build_phase, dict):
            build_phase["source_rows"] = "Rows where has_deck is true after joining actual game_data decks."
            build_phase["target_deck_counts"] = DECK_COUNTS_KEY
            build_phase["spell_target_deck_counts"] = DECK_SPELL_COUNTS_KEY
            build_phase["stop_condition"] = "deck_counts == target_deck_counts; actual deck sizes are not forced to 40."

    save_json(data_dir / "metadata.json", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace cleaned deck targets with actual game_data decks.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--draft-csv", type=Path, default=DEFAULT_DRAFT_CSV)
    parser.add_argument("--game-csv", type=Path, default=DEFAULT_GAME_CSV)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--game-chunksize", type=int, default=50_000)
    parser.add_argument("--min-match-wins", type=int, default=5)
    parser.add_argument("--complete-pool-size", type=int, default=42)
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping missing/invalid game deck targets.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    draft_csv = args.draft_csv.resolve()
    game_csv = args.game_csv.resolve()

    if not draft_csv.exists():
        raise FileNotFoundError(f"Draft CSV not found: {draft_csv}")
    if not game_csv.exists():
        raise FileNotFoundError(f"Game CSV not found: {game_csv}")

    metadata = load_json(data_dir / "metadata.json")
    card_names = validate_metadata(metadata)

    terminal_rows = collect_terminal_rows(
        data_dir=data_dir,
        draft_csv=draft_csv,
        card_count=len(card_names),
        chunksize=args.chunksize,
        min_match_wins=args.min_match_wins,
        complete_pool_size=args.complete_pool_size,
    )
    relevant_draft_ids = {row.draft_id for row in terminal_rows}
    print(f"Recovered {len(terminal_rows):,} terminal cleaned rows ({len(relevant_draft_ids):,} draft_ids).")

    selected_decks, build_count_distribution = collect_latest_game_decks(
        game_csv=game_csv,
        card_names=card_names,
        relevant_draft_ids=relevant_draft_ids,
        chunksize=args.game_chunksize,
    )
    print(f"Found actual game decks for {len(selected_decks):,} terminal draft_ids.")

    replacement_stats = replace_chunk_targets(
        data_dir=data_dir,
        terminal_rows=terminal_rows,
        selected_decks=selected_decks,
        card_names=card_names,
        strict=args.strict,
    )
    update_metadata(
        data_dir=data_dir,
        metadata=metadata,
        draft_csv=draft_csv,
        game_csv=game_csv,
        chunksize=args.chunksize,
        min_match_wins=args.min_match_wins,
        complete_pool_size=args.complete_pool_size,
        build_count_distribution=build_count_distribution,
        replacement_stats=replacement_stats,
    )

    print(f"Wrote actual deck targets to {data_dir}")
    print(
        "Rows with actual decks: "
        f"{replacement_stats['rows_with_decks']:,} / {replacement_stats['terminal_rows']:,}; "
        f"missing game rows {replacement_stats['missing_game_rows']:,}; "
        f"invalid pool rows {replacement_stats['invalid_pool_rows']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
