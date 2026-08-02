#!/usr/bin/env python3
"""Train a draft-pick model from cleaned 17Lands-style pick chunks."""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from NN import ContextualDraftScorer, DraftScorer


# ---- config: edit these, then run `python train.py` -------------------------
ROOT = Path(__file__).resolve().parent
DECK_DATA_DIR = ROOT / "data" / "cleaned_with_decks"
DATA_DIR = DECK_DATA_DIR if DECK_DATA_DIR.exists() else ROOT / "data" / "cleaned"
CARD_DATA = ROOT / "data" / "raw" / "SOS_cards.json"
EXTRA_CARD_DATA = ROOT / "data" / "raw" / "SOS_extra_cards.json"
OUTPUT_DIR = ROOT / "models"
TOP1_PLOT = OUTPUT_DIR / "top1_over_epochs.png"

MODEL = "draftscorer"          # "contextual" or "draftscorer"
EPOCHS = 10
BATCH_SIZE = 2048
LR = 2e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.02
VAL_FRACTION = 0.08
SEED = 42

EMB_DIM = 128
HIDDEN_DIM = 256
SYNERGY_DIM = 64
DROPOUT = 0.15
GRAD_CLIP = 1.0
TARGET_DECK_SIZE = 40
MAX_BUILD_STEPS = TARGET_DECK_SIZE
MAX_POOL_SIZE = 45
MAX_PACK_SIZE = 15

DEVICE = "cuda"               # "auto", "cpu", "cuda", etc.
NUM_THREADS = 0               # 0 keeps PyTorch default
USE_AMP = False
COMPILE = False
RESUME: Path | None = None

LIMIT_CHUNKS: int | None = None
MAX_STEPS_PER_EPOCH: int | None = None
MAX_VAL_BATCHES: int | None = None
LOG_EVERY = 100
SAVE_EVERY_EPOCH = False
VALIDATE_BUILD_ROLLOUT = True
ROLLOUT_VAL_DECK_LIMIT: int | None = None
TRAIN_BUILD_ROLLOUT = True
ROLLOUT_TRAIN_START_EPOCH = 0
ROLLOUT_TRAIN_DECKS_PER_CHUNK: int | None = None
ROLLOUT_TRAIN_MAX_STEPS: int | None = TARGET_DECK_SIZE
ROLLOUT_TRAIN_BATCH_DECKS = 256
USE_CARD_FEATURES = True
CHOICE_WEIGHTING = True
TRAIN_DRAFT_PHASE = True
TRAIN_BUILD_PHASE = True
BASIC_LAND_NAMES = ("Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes")
DECK_TARGET_KEY = "deck_counts"
HAS_DECK_KEY = "has_deck"

QUICK = False                 # one short epoch on a few chunks
QUICK_CHUNKS = 3
QUICK_TRAIN_BATCHES = 20
QUICK_VAL_BATCHES = 10
# ----------------------------------------------------------------------------


class Metrics:
    def __init__(self) -> None:
        self.loss_sum = self.samples = 0
        self.draft_samples = self.draft_correct = self.draft_top3_correct = 0
        self.draft_mrr_sum = self.draft_nontrivial_samples = self.draft_nontrivial_correct = 0
        self.deckbuilding_samples = self.deckbuilding_set_correct = self.deckbuilding_exact_correct = 0

    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor | None,
        mask: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
    ) -> None:
        n = logits.shape[0]
        pred = logits.argmax(1)
        if target_mask is None:
            if targets is None:
                raise ValueError("targets are required when target_mask is None.")
            correct = pred == targets
            exact_correct = correct
            top3_correct = (logits.topk(min(3, logits.shape[1]), 1).indices == targets[:, None]).any(1)
            target_logits = logits.gather(1, targets[:, None])
        else:
            correct = target_mask.gather(1, pred[:, None]).squeeze(1)
            exact_correct = correct if targets is None else pred == targets
            top3_correct = target_mask.gather(1, logits.topk(min(3, logits.shape[1]), 1).indices).any(1)
            target_logits = logits.masked_fill(~target_mask, -1e9).max(1, keepdim=True).values
        hard = mask.sum(1) > 1
        draft = torch.ones(n, dtype=torch.bool, device=logits.device) if phase is None else phase <= 0.5
        deckbuilding = ~draft
        reciprocal_rank = 1.0 / ((logits > target_logits).sum(1).float() + 1.0)

        self.loss_sum += float(loss.detach().cpu()) * n
        self.samples += n

        draft_count = int(draft.sum().detach().cpu())
        if draft_count:
            draft_hard = draft & hard
            self.draft_samples += draft_count
            self.draft_correct += int(correct[draft].sum().detach().cpu())
            self.draft_top3_correct += int(top3_correct[draft].sum().detach().cpu())
            self.draft_mrr_sum += float(reciprocal_rank[draft].sum().detach().cpu())
            self.draft_nontrivial_samples += int(draft_hard.sum().detach().cpu())
            self.draft_nontrivial_correct += int(correct[draft_hard].sum().detach().cpu()) if bool(draft_hard.any()) else 0

        deckbuilding_count = int(deckbuilding.sum().detach().cpu())
        if deckbuilding_count:
            self.deckbuilding_samples += deckbuilding_count
            self.deckbuilding_set_correct += int(correct[deckbuilding].sum().detach().cpu())
            self.deckbuilding_exact_correct += int(exact_correct[deckbuilding].sum().detach().cpu())

    def as_dict(self) -> dict[str, float]:
        n = max(self.samples, 1)
        d = max(self.draft_samples, 1)
        h = max(self.draft_nontrivial_samples, 1)
        b = max(self.deckbuilding_samples, 1)
        return {
            "loss": self.loss_sum / n,
            "top1": self.draft_correct / d,
            "top3": self.draft_top3_correct / d,
            "mrr": self.draft_mrr_sum / d,
            "nontrivial_top1": self.draft_nontrivial_correct / h,
            "deckbuilding_set_accuracy": self.deckbuilding_set_correct / b,
            "deckbuilding_exact_accuracy": self.deckbuilding_exact_correct / b,
            "deckbuilding_accuracy": self.deckbuilding_set_correct / b,
        }


def config_dict() -> dict[str, Any]:
    keys = [
        "DATA_DIR", "CARD_DATA", "EXTRA_CARD_DATA", "OUTPUT_DIR", "MODEL", "EPOCHS", "BATCH_SIZE", "LR",
        "WEIGHT_DECAY", "LABEL_SMOOTHING", "VAL_FRACTION", "SEED", "EMB_DIM",
        "HIDDEN_DIM", "SYNERGY_DIM", "DROPOUT", "GRAD_CLIP", "DEVICE", "NUM_THREADS",
        "USE_AMP", "COMPILE", "RESUME", "LIMIT_CHUNKS", "MAX_STEPS_PER_EPOCH",
        "MAX_VAL_BATCHES", "LOG_EVERY", "SAVE_EVERY_EPOCH", "VALIDATE_BUILD_ROLLOUT",
        "ROLLOUT_VAL_DECK_LIMIT", "TRAIN_BUILD_ROLLOUT", "ROLLOUT_TRAIN_START_EPOCH",
        "ROLLOUT_TRAIN_DECKS_PER_CHUNK", "ROLLOUT_TRAIN_MAX_STEPS", "ROLLOUT_TRAIN_BATCH_DECKS",
        "USE_CARD_FEATURES",
        "CHOICE_WEIGHTING", "TRAIN_DRAFT_PHASE", "TRAIN_BUILD_PHASE", "MAX_BUILD_STEPS",
        "MAX_POOL_SIZE", "MAX_PACK_SIZE", "TARGET_DECK_SIZE", "QUICK", "QUICK_CHUNKS",
        "QUICK_TRAIN_BATCHES", "QUICK_VAL_BATCHES", "TOP1_PLOT",
    ]
    return {key: str(value) if isinstance((value := globals()[key]), Path) else value for key in keys}


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def draft_chunk_paths(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.glob("draft_xy_chunk_*.npz")
        if path.stem.removeprefix("draft_xy_chunk_").isdigit()
    )


def parse_number(value: Any) -> float:
    try:
        return 0.0 if value is None else float(value)
    except (TypeError, ValueError):
        return 0.0


def build_card_features(card_names: list[str]) -> tuple[torch.Tensor, list[str], dict[str, int]]:
    if not USE_CARD_FEATURES or not CARD_DATA.exists():
        return torch.empty(len(card_names), 0), [], {"matched": 0, "missing": len(card_names)}

    lookup: dict[str, dict[str, Any]] = {}
    card_files = [CARD_DATA]
    if EXTRA_CARD_DATA.exists():
        card_files.append(EXTRA_CARD_DATA)

    for card_file in card_files:
        for card in load_json(card_file):
            if name := card.get("name"):
                lookup.setdefault(name, card)
                for part in name.split(" // "):
                    lookup.setdefault(part, card)

    colors = ["W", "U", "B", "R", "G"]
    types = ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Land", "Planeswalker", "Battle"]
    rarities = ["common", "uncommon", "rare", "mythic"]
    keywords = [
        "Flying", "Trample", "Vigilance", "Haste", "Reach", "Deathtouch", "Lifelink",
        "Menace", "Ward", "Prowess", "Flash", "First strike", "Double strike", "Converge",
    ]
    feature_names = (
        [f"color_{c}" for c in colors]
        + ["is_colorless", "is_multicolor"]
        + [f"type_{t.lower()}" for t in types]
        + [f"rarity_{r}" for r in rarities]
        + [f"keyword_{k.lower().replace(' ', '_')}" for k in keywords]
        + ["cmc_scaled", "power_scaled", "toughness_scaled", "oracle_len_scaled", "keyword_count_scaled", "metadata_missing"]
    )

    rows, matched = [], 0
    for card_name in card_names:
        card = lookup.get(card_name) or {}
        missing = not card
        matched += 0 if missing else 1
        card_colors = set(card.get("colors") or [])
        card_keywords = set(card.get("keywords") or [])
        oracle = card.get("oracle_text") or ""
        type_line = card.get("type_line") or ""

        rows.append(
            [float(c in card_colors) for c in colors]
            + [float(not card_colors and not missing), float(len(card_colors) > 1)]
            + [float(t in type_line) for t in types]
            + [float(card.get("rarity") == r) for r in rarities]
            + [float(k in card_keywords) for k in keywords]
            + [
                min(parse_number(card.get("cmc")), 12.0) / 12.0,
                min(parse_number(card.get("power")), 12.0) / 12.0,
                min(parse_number(card.get("toughness")), 12.0) / 12.0,
                min(len(oracle), 500) / 500.0,
                min(len(card_keywords), 8) / 8.0,
                float(missing),
            ]
        )

    return torch.tensor(rows, dtype=torch.float32), feature_names, {"matched": matched, "missing": len(card_names) - matched}


def validation_mask(rows: int, chunk_index: int, val_fraction: float) -> np.ndarray:
    return np.random.default_rng(SEED + chunk_index * 1_000_003).random(rows) < val_fraction


def has_deck_targets(path: Path) -> bool:
    with np.load(path) as data:
        return {DECK_TARGET_KEY, HAS_DECK_KEY}.issubset(data.files)


def should_train_build(chunks: list[Path]) -> bool:
    return MODEL == "draftscorer" and TRAIN_BUILD_PHASE and bool(chunks) and has_deck_targets(chunks[0])


def basic_land_mask(card_names: list[str]) -> np.ndarray:
    return np.asarray([card_name in BASIC_LAND_NAMES for card_name in card_names], dtype=np.bool_)


def build_example_count(deck_counts: np.ndarray, has_deck: np.ndarray, rows: np.ndarray) -> int:
    if rows.size == 0:
        return 0
    deck_rows = rows[has_deck[rows]]
    if deck_rows.size == 0:
        return 0
    return int(deck_counts[deck_rows].sum())


def rollout_example_count(deck_counts: np.ndarray, has_deck: np.ndarray, rows: np.ndarray) -> int:
    if rows.size == 0 or not TRAIN_BUILD_ROLLOUT:
        return 0
    deck_rows = rows[has_deck[rows]]
    if deck_rows.size == 0:
        return 0
    if ROLLOUT_TRAIN_DECKS_PER_CHUNK is not None:
        deck_rows = deck_rows[:ROLLOUT_TRAIN_DECKS_PER_CHUNK]
    target_sizes = deck_counts[deck_rows].sum(axis=1).astype(np.int64, copy=False)
    if ROLLOUT_TRAIN_MAX_STEPS is not None:
        target_sizes = np.minimum(target_sizes, int(ROLLOUT_TRAIN_MAX_STEPS))
    return int(target_sizes.clip(min=0).sum())


def split_summary(chunks: list[Path], cards: int, val_fraction: float, batch_size: int, train_build: bool) -> tuple[int, int, int, int]:
    train = val = train_batches = val_batches = 0
    for chunk_index, path in enumerate(chunks):
        with np.load(path) as data:
            x, y = data["X"], data["y"]
            mask = validation_mask(len(y), chunk_index, val_fraction) if val_fraction > 0 else np.zeros(len(y), dtype=bool)
            train_rows = np.flatnonzero(~mask)
            val_rows = np.flatnonzero(mask)

            if TRAIN_DRAFT_PHASE:
                train += int(train_rows.size)
                val += int(val_rows.size)
                train_batches += math.ceil(train_rows.size / batch_size) if train_rows.size else 0
                val_batches += math.ceil(val_rows.size / batch_size) if val_rows.size else 0

            if train_build and {DECK_TARGET_KEY, HAS_DECK_KEY}.issubset(data.files):
                train_build_examples = build_example_count(data[DECK_TARGET_KEY], data[HAS_DECK_KEY], train_rows)
                val_build_examples = build_example_count(data[DECK_TARGET_KEY], data[HAS_DECK_KEY], val_rows)
                train += train_build_examples
                val += val_build_examples
                train_batches += math.ceil(train_build_examples / batch_size) if train_build_examples else 0
                val_batches += math.ceil(val_build_examples / batch_size) if val_build_examples else 0
                train_rollout_examples = rollout_example_count(data[DECK_TARGET_KEY], data[HAS_DECK_KEY], train_rows)
                train += train_rollout_examples
                train_batches += math.ceil(train_rollout_examples / batch_size) if train_rollout_examples else 0

    return train, val, train_batches, val_batches


def make_draft_batch(x: np.ndarray, y: np.ndarray, rows: np.ndarray, cards: int) -> dict[str, np.ndarray]:
    batch_x = x[rows, : 2 + 2 * cards]
    n = rows.size
    return {
        "pool": np.ascontiguousarray(batch_x[:, 2 + cards : 2 + 2 * cards]),
        "pack": np.ascontiguousarray(batch_x[:, 2 : 2 + cards]),
        "deck": np.zeros((n, cards), dtype=np.uint8),
        "pack_no": np.ascontiguousarray(batch_x[:, 0]),
        "pick_no": np.ascontiguousarray(batch_x[:, 1]),
        "phase": np.zeros(n, dtype=np.uint8),
        "build_step": np.zeros(n, dtype=np.uint8),
        "targets": np.ascontiguousarray(y[rows]).astype(np.int64, copy=False),
        "target_mask": None,
    }


def make_build_examples(
    x: np.ndarray,
    y: np.ndarray,
    deck_counts: np.ndarray,
    has_deck: np.ndarray,
    rows: np.ndarray,
    cards: int,
) -> dict[str, np.ndarray] | None:
    pool_rows, deck_rows, pack_nos, pick_nos, build_steps, targets, target_masks = [], [], [], [], [], [], []
    pool_start = 2 + cards
    pool_stop = pool_start + cards

    for row in rows[has_deck[rows]]:
        pool = x[row, pool_start:pool_stop].astype(np.uint8, copy=True)
        pool[int(y[row])] += 1
        target = deck_counts[row].astype(np.uint8, copy=False)
        deck = np.zeros(cards, dtype=np.uint8)
        step = 0

        while np.any(deck < target):
            valid_adds = deck < target
            pool_rows.append(pool.copy())
            deck_rows.append(deck.copy())
            pack_nos.append(x[row, 0])
            pick_nos.append(x[row, 1])
            build_steps.append(step)

            add = int(np.flatnonzero(valid_adds)[0])
            targets.append(add)
            target_masks.append(valid_adds.copy())
            deck[add] += 1
            step += 1

    if not pool_rows:
        return None

    n = len(pool_rows)
    return {
        "pool": np.ascontiguousarray(np.stack(pool_rows)),
        "pack": np.zeros((n, cards), dtype=np.uint8),
        "deck": np.ascontiguousarray(np.stack(deck_rows)),
        "pack_no": np.asarray(pack_nos, dtype=np.uint8),
        "pick_no": np.asarray(pick_nos, dtype=np.uint8),
        "phase": np.ones(n, dtype=np.uint8),
        "build_step": np.asarray(build_steps, dtype=np.uint8),
        "targets": np.asarray(targets, dtype=np.int64),
        "target_mask": np.ascontiguousarray(np.stack(target_masks)),
    }


def make_rollout_build_examples(
    model: nn.Module,
    device: torch.device,
    x: np.ndarray,
    y: np.ndarray,
    deck_counts: np.ndarray,
    has_deck: np.ndarray,
    rows: np.ndarray,
    cards: int,
    basic_land_mask_tensor: torch.Tensor,
    rng: np.random.Generator,
) -> dict[str, np.ndarray] | None:
    deck_rows = rows[has_deck[rows]]
    if deck_rows.size == 0:
        return None
    if ROLLOUT_TRAIN_DECKS_PER_CHUNK is not None and deck_rows.size > ROLLOUT_TRAIN_DECKS_PER_CHUNK:
        deck_rows = rng.choice(deck_rows, size=ROLLOUT_TRAIN_DECKS_PER_CHUNK, replace=False)

    pool_rows, built_rows, pack_nos, pick_nos, build_steps, targets, target_masks = [], [], [], [], [], [], []
    pool_start = 2 + cards
    pool_stop = pool_start + cards
    batch_decks = max(1, int(ROLLOUT_TRAIN_BATCH_DECKS))
    max_config_steps = None if ROLLOUT_TRAIN_MAX_STEPS is None else int(ROLLOUT_TRAIN_MAX_STEPS)
    basic_mask_np = basic_land_mask_tensor.detach().cpu().numpy().astype(np.bool_, copy=False)
    if basic_mask_np.ndim != 1:
        basic_mask_np = basic_mask_np.reshape(-1)

    was_training = model.training
    model.eval()
    try:
        for batch_start in range(0, deck_rows.size, batch_decks):
            batch_rows = deck_rows[batch_start : batch_start + batch_decks]
            pools, real_targets, row_pack_nos, row_pick_nos, target_sizes = [], [], [], [], []

            for row in batch_rows:
                pool = x[row, pool_start:pool_stop].astype(np.uint8, copy=True)
                pool[int(y[row])] += 1
                target = deck_counts[row].astype(np.int16, copy=False)
                target_size = int(target.sum())
                if max_config_steps is not None:
                    target_size = min(target_size, max_config_steps)
                if target_size <= 0:
                    continue

                pools.append(pool)
                real_targets.append(target)
                row_pack_nos.append(x[row, 0])
                row_pick_nos.append(x[row, 1])
                target_sizes.append(target_size)

            if not pools:
                continue

            pools_np = np.stack(pools).astype(np.int16, copy=False)
            targets_np = np.stack(real_targets).astype(np.int16, copy=False)
            built_np = np.zeros((len(pools), cards), dtype=np.int16)
            row_pack_nos_np = np.asarray(row_pack_nos, dtype=np.uint8)
            row_pick_nos_np = np.asarray(row_pick_nos, dtype=np.uint8)

            for step in range(max(target_sizes)):
                active = np.asarray([step < size for size in target_sizes], dtype=np.bool_)
                if not active.any():
                    break
                active_rows = np.flatnonzero(active)
                active_built = built_np[active_rows]
                legal_np = (pools_np[active_rows] > active_built) | basic_mask_np.reshape(1, -1)
                remaining_target_np = (targets_np[active_rows] > active_built) & legal_np
                usable = remaining_target_np.any(axis=1)

                if usable.any():
                    usable_rows = active_rows[usable]
                    usable_targets = remaining_target_np[usable]
                    pool_rows.extend(pools_np[usable_rows].astype(np.uint8, copy=False))
                    built_rows.extend(built_np[usable_rows].astype(np.uint8, copy=False))
                    pack_nos.extend(row_pack_nos_np[usable_rows])
                    pick_nos.extend(row_pick_nos_np[usable_rows])
                    build_steps.extend([step] * usable_rows.size)
                    targets.extend(np.argmax(usable_targets, axis=1).astype(np.int64).tolist())
                    target_masks.extend(usable_targets.copy())

                pool_t = torch.as_tensor(pools_np[active_rows], dtype=torch.float32, device=device)
                built_t = torch.as_tensor(active_built, dtype=torch.float32, device=device)
                pack_t = torch.zeros((active_rows.size, cards), dtype=torch.float32, device=device)
                pack_no_t = torch.as_tensor(row_pack_nos_np[active_rows], dtype=torch.long, device=device)
                pick_no_t = torch.as_tensor(row_pick_nos_np[active_rows], dtype=torch.long, device=device)
                phase_t = torch.ones(active_rows.size, dtype=torch.float32, device=device)
                step_t = torch.full((active_rows.size,), float(step), dtype=torch.float32, device=device)
                legal_t = (pool_t > built_t) | basic_land_mask_tensor.view(1, -1)

                with torch.no_grad():
                    preds = logits_for(
                        model,
                        {
                            "pool": pool_t,
                            "pack": pack_t,
                            "deck": built_t,
                            "pack_no": pack_no_t,
                            "pick_no": pick_no_t,
                            "phase": phase_t,
                            "build_step": step_t,
                        },
                        legal_t,
                    ).argmax(1).detach().cpu().numpy()

                for local_index, pred in enumerate(preds):
                    built_np[active_rows[local_index], int(pred)] += 1
    finally:
        model.train(was_training)

    if not pool_rows:
        return None

    n = len(pool_rows)
    return {
        "pool": np.ascontiguousarray(np.stack(pool_rows).astype(np.uint8, copy=False)),
        "pack": np.zeros((n, cards), dtype=np.uint8),
        "deck": np.ascontiguousarray(np.stack(built_rows).astype(np.uint8, copy=False)),
        "pack_no": np.asarray(pack_nos, dtype=np.uint8),
        "pick_no": np.asarray(pick_nos, dtype=np.uint8),
        "phase": np.ones(n, dtype=np.uint8),
        "build_step": np.asarray(build_steps, dtype=np.uint8),
        "targets": np.asarray(targets, dtype=np.int64),
        "target_mask": np.ascontiguousarray(np.stack(target_masks)),
    }


def iter_phase_batches(
    chunks: list[Path],
    cards: int,
    batch_size: int,
    val_fraction: float,
    epoch: int,
    split: str,
    max_batches: int | None,
    train_build: bool,
    rollout_model: nn.Module | None = None,
    rollout_device: torch.device | None = None,
    basic_land_mask_tensor: torch.Tensor | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    rng = np.random.default_rng(SEED + epoch * 10_007 + (0 if split == "train" else 50_000))
    order = list(enumerate(chunks))
    if split == "train":
        rng.shuffle(order)

    seen = 0
    for chunk_index, path in order:
        with np.load(path) as data:
            x, y = data["X"], data["y"]
            mask = validation_mask(len(y), chunk_index, val_fraction) if val_fraction > 0 else np.zeros(len(y), dtype=bool)
            rows = np.flatnonzero(~mask if split == "train" else mask)
            if split == "train":
                rng.shuffle(rows)

            batches: list[dict[str, np.ndarray]] = []
            if TRAIN_DRAFT_PHASE and rows.size:
                for start in range(0, len(rows), batch_size):
                    batches.append(make_draft_batch(x, y, rows[start : start + batch_size], cards))

            if train_build and {DECK_TARGET_KEY, HAS_DECK_KEY}.issubset(data.files):
                build_examples = make_build_examples(x, y, data[DECK_TARGET_KEY], data[HAS_DECK_KEY], rows, cards)
                if build_examples is not None:
                    build_rows = np.arange(build_examples["pool"].shape[0])
                    if split == "train":
                        rng.shuffle(build_rows)
                    for start in range(0, len(build_rows), batch_size):
                        batch_rows = build_rows[start : start + batch_size]
                        batches.append({
                            key: (None if value is None else np.ascontiguousarray(value[batch_rows]))
                            for key, value in build_examples.items()
                        })

                rollout_build = (
                    split == "train"
                    and TRAIN_BUILD_ROLLOUT
                    and epoch >= ROLLOUT_TRAIN_START_EPOCH
                    and rollout_model is not None
                    and rollout_device is not None
                    and basic_land_mask_tensor is not None
                )
                if rollout_build:
                    rollout_examples = make_rollout_build_examples(
                        model=rollout_model,
                        device=rollout_device,
                        x=x,
                        y=y,
                        deck_counts=data[DECK_TARGET_KEY],
                        has_deck=data[HAS_DECK_KEY],
                        rows=rows,
                        cards=cards,
                        basic_land_mask_tensor=basic_land_mask_tensor,
                        rng=rng,
                    )
                    if rollout_examples is not None:
                        rollout_rows = np.arange(rollout_examples["pool"].shape[0])
                        rng.shuffle(rollout_rows)
                        for start in range(0, len(rollout_rows), batch_size):
                            batch_rows = rollout_rows[start : start + batch_size]
                            batches.append({
                                key: (None if value is None else np.ascontiguousarray(value[batch_rows]))
                                for key, value in rollout_examples.items()
                            })

            if split == "train":
                rng.shuffle(batches)

            for batch in batches:
                if max_batches is not None and seen >= max_batches:
                    return
                seen += 1
                yield batch


def unpack_batch(
    batch: dict[str, np.ndarray],
    device: torch.device,
    basic_land_mask_tensor: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    tensors = {
        "pool": torch.as_tensor(batch["pool"], dtype=torch.float32, device=device),
        "pack": torch.as_tensor(batch["pack"], dtype=torch.float32, device=device),
        "deck": torch.as_tensor(batch["deck"], dtype=torch.float32, device=device),
        "pack_no": torch.as_tensor(batch["pack_no"], dtype=torch.long, device=device),
        "pick_no": torch.as_tensor(batch["pick_no"], dtype=torch.long, device=device),
        "phase": torch.as_tensor(batch["phase"], dtype=torch.float32, device=device),
        "build_step": torch.as_tensor(batch["build_step"], dtype=torch.float32, device=device),
    }
    targets = None if batch["targets"] is None else torch.as_tensor(batch["targets"], dtype=torch.long, device=device)
    target_mask = None if batch["target_mask"] is None else torch.as_tensor(batch["target_mask"], dtype=torch.bool, device=device)
    build_mask = (tensors["pool"] > tensors["deck"]) | basic_land_mask_tensor.view(1, -1)
    legal_mask = torch.where(tensors["phase"][:, None] > 0.5, build_mask, tensors["pack"] > 0)
    return tensors, targets, target_mask, legal_mask


def logits_for(model: nn.Module, inputs: dict[str, torch.Tensor], legal_mask: torch.Tensor) -> torch.Tensor:
    if MODEL == "draftscorer":
        return model(
            inputs["pool"],
            inputs["pack"],
            inputs["pack_no"],
            inputs["pick_no"],
            deck_counts=inputs["deck"],
            phases=inputs["phase"],
            build_steps=inputs["build_step"],
            legal_mask=legal_mask,
        )
    return model(inputs["pool"], inputs["pack"], inputs["pack_no"], inputs["pick_no"])


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor | None,
    mask: torch.Tensor,
    smoothing: float,
    choice_weighting: bool,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, 1)
    if target_mask is None:
        if targets is None:
            raise ValueError("targets are required when target_mask is None.")
        per_row = -log_probs.gather(1, targets[:, None]).squeeze(1)
    else:
        target_log_probs = log_probs.masked_fill(~target_mask, -1e9)
        per_row = -torch.logsumexp(target_log_probs, dim=1)

    if smoothing > 0:
        legal = mask.float()
        per_row = (1.0 - smoothing) * per_row + smoothing * (-(log_probs * legal).sum(1) / legal.sum(1).clamp_min(1.0))
    if not choice_weighting:
        return per_row.mean()
    weights = torch.log2(mask.sum(1).float().clamp_min(2.0))
    return (per_row * weights).sum() / weights.sum().clamp_min(1.0)


def amp_context(device: torch.device, enabled: bool):
    return torch.amp.autocast("cuda") if enabled and device.type == "cuda" else nullcontext()


def run_epoch(
    model: nn.Module,
    opt: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    chunks: list[Path],
    cards: int,
    epoch: int,
    batch_size: int,
    val_fraction: float,
    max_batches: int | None,
    train_build: bool,
    basic_land_mask_tensor: torch.Tensor,
) -> dict[str, float]:
    training = opt is not None
    model.train(training)
    metrics, start = Metrics(), time.time()
    split = "train" if training else "val"

    for step, batch in enumerate(
        iter_phase_batches(
            chunks,
            cards,
            batch_size,
            val_fraction,
            epoch,
            split,
            max_batches,
            train_build,
            rollout_model=model if training else None,
            rollout_device=device if training else None,
            basic_land_mask_tensor=basic_land_mask_tensor if training else None,
        ),
        1,
    ):
        inputs, targets, target_mask, mask = unpack_batch(batch, device, basic_land_mask_tensor)
        with torch.set_grad_enabled(training), amp_context(device, USE_AMP and training):
            logits = logits_for(model, inputs, mask)
            loss = masked_cross_entropy(
                logits,
                targets,
                mask,
                LABEL_SMOOTHING if training else 0.0,
                CHOICE_WEIGHTING if training else False,
                target_mask,
            )

        if training:
            opt.zero_grad(set_to_none=True)
            if scaler is None:
                loss.backward()
                if GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            else:
                scaler.scale(loss).backward()
                if GRAD_CLIP > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            scheduler.step()

        metrics.update(loss, logits.detach(), targets, mask, target_mask, inputs["phase"])
        if training and LOG_EVERY > 0 and step % LOG_EVERY == 0:
            now = metrics.as_dict()
            speed = metrics.samples / max(time.time() - start, 1e-6)
            print(
                f"  step {step:5d} | loss {now['loss']:.4f} | top1 {now['top1']:.4f} | "
                f"deckset {now['deckbuilding_set_accuracy']:.4f} | "
                f"deckexact {now['deckbuilding_exact_accuracy']:.4f} | "
                f"nontriv {now['nontrivial_top1']:.4f} | lr {opt.param_groups[0]['lr']:.2e} | "
                f"{speed:,.0f} rows/s"
            )

    result = metrics.as_dict()
    result["seconds"] = time.time() - start
    return result


def rollout_validation_rows(path: Path, chunk_index: int, cards: int, val_fraction: float) -> tuple[np.ndarray, np.lib.npyio.NpzFile]:
    data = np.load(path)
    mask = validation_mask(len(data["y"]), chunk_index, val_fraction) if val_fraction > 0 else np.ones(len(data["y"]), dtype=bool)
    rows = np.flatnonzero(mask & data[HAS_DECK_KEY])
    return rows, data


def evaluate_build_rollout(
    model: nn.Module,
    device: torch.device,
    chunks: list[Path],
    cards: int,
    val_fraction: float,
    basic_land_mask_tensor: torch.Tensor,
    max_decks: int | None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    exact_decks = decks_seen = overlap_cards = target_cards = predicted_cards = 0
    pool_start = 2 + cards
    pool_stop = pool_start + cards
    batch_decks = 256

    try:
        for chunk_index, path in enumerate(chunks):
            rows, data = rollout_validation_rows(path, chunk_index, cards, val_fraction)
            try:
                if max_decks is not None:
                    remaining = max_decks - decks_seen
                    if remaining <= 0:
                        break
                    rows = rows[:remaining]
                if rows.size == 0:
                    continue

                X, y, target_decks = data["X"], data["y"], data[DECK_TARGET_KEY]
                for start in range(0, rows.size, batch_decks):
                    batch_rows = rows[start : start + batch_decks]
                    pools, targets, pack_nos, pick_nos, target_sizes = [], [], [], [], []
                    for row in batch_rows:
                        pool = X[row, pool_start:pool_stop].astype(np.uint8, copy=True)
                        pool[int(y[row])] += 1
                        target = target_decks[row].astype(np.int16, copy=False)
                        pools.append(pool)
                        targets.append(target)
                        pack_nos.append(X[row, 0])
                        pick_nos.append(X[row, 1])
                        target_sizes.append(int(target.sum()))

                    pools_np = np.stack(pools)
                    built_np = np.zeros((len(batch_rows), cards), dtype=np.int16)
                    max_steps = max(target_sizes)

                    for step in range(max_steps):
                        active = np.asarray([step < size for size in target_sizes], dtype=np.bool_)
                        if not active.any():
                            break
                        active_rows = np.flatnonzero(active)
                        pool_t = torch.as_tensor(pools_np[active_rows], dtype=torch.float32, device=device)
                        built_t = torch.as_tensor(built_np[active_rows], dtype=torch.float32, device=device)
                        pack_t = torch.zeros((active_rows.size, cards), dtype=torch.float32, device=device)
                        pack_no_t = torch.as_tensor(np.asarray(pack_nos)[active_rows], dtype=torch.long, device=device)
                        pick_no_t = torch.as_tensor(np.asarray(pick_nos)[active_rows], dtype=torch.long, device=device)
                        phase_t = torch.ones(active_rows.size, dtype=torch.float32, device=device)
                        step_t = torch.full((active_rows.size,), float(step), dtype=torch.float32, device=device)
                        legal_mask = (pool_t > built_t) | basic_land_mask_tensor.view(1, -1)

                        with torch.no_grad():
                            preds = logits_for(
                                model,
                                {
                                    "pool": pool_t,
                                    "pack": pack_t,
                                    "deck": built_t,
                                    "pack_no": pack_no_t,
                                    "pick_no": pick_no_t,
                                    "phase": phase_t,
                                    "build_step": step_t,
                                },
                                legal_mask,
                            ).argmax(1).detach().cpu().numpy()

                        for local_index, pred in enumerate(preds):
                            built_np[active_rows[local_index], int(pred)] += 1

                    for built, target in zip(built_np, targets):
                        exact_decks += int(np.array_equal(built, target))
                        overlap = int(np.minimum(built, target).sum())
                        overlap_cards += overlap
                        target_cards += int(target.sum())
                        predicted_cards += int(built.sum())
                        decks_seen += 1
            finally:
                data.close()
    finally:
        model.train(was_training)

    if decks_seen == 0:
        return {
            "deckbuild_rollout_decks": 0.0,
            "deckbuild_rollout_exact": 0.0,
            "deckbuild_rollout_overlap": 0.0,
            "deckbuild_rollout_precision": 0.0,
        }
    return {
        "deckbuild_rollout_decks": float(decks_seen),
        "deckbuild_rollout_exact": exact_decks / decks_seen,
        "deckbuild_rollout_overlap": overlap_cards / max(target_cards, 1),
        "deckbuild_rollout_precision": overlap_cards / max(predicted_cards, 1),
    }


def make_model(cards: int, features: torch.Tensor, basic_land_indices: list[int]) -> tuple[nn.Module, dict[str, Any]]:
    if MODEL == "draftscorer":
        kwargs = {
            "num_cards": cards,
            "emb_dim": EMB_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "max_build_steps": MAX_BUILD_STEPS,
            "max_pool_size": MAX_POOL_SIZE,
            "max_pack_size": MAX_PACK_SIZE,
            "target_deck_size": TARGET_DECK_SIZE,
            "basic_land_indices": basic_land_indices,
        }
        return DraftScorer(**kwargs), kwargs
    kwargs = {
        "num_cards": cards,
        "card_features": features,
        "emb_dim": EMB_DIM,
        "hidden_dim": HIDDEN_DIM,
        "synergy_dim": SYNERGY_DIM,
        "dropout": DROPOUT,
    }
    return ContextualDraftScorer(**kwargs), kwargs


def uncompiled(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def save_checkpoint(path: Path, model: nn.Module, opt: torch.optim.Optimizer, scheduler: Any, epoch: int, best: float, metadata: dict[str, Any], model_kwargs: dict[str, Any], history: list[dict[str, Any]], feature_names: list[str], feature_match: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best,
            "model_name": MODEL,
            "model_kwargs": model_kwargs,
            "model_state": uncompiled(model).state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "args": config_dict(),
            "history": history,
            "card_names": metadata["card_names"],
            "feature_names": feature_names,
            "feature_match": feature_match,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, opt: torch.optim.Optimizer, scheduler: Any, device: torch.device) -> tuple[int, float, list[dict[str, Any]]]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    opt.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_val_loss", math.inf)), checkpoint.get("history", [])


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    parts = [
        f"{prefix} loss {metrics['loss']:.4f}",
        f"top1 {metrics['top1']:.4f}",
        f"top3 {metrics['top3']:.4f}",
        f"deckset {metrics['deckbuilding_set_accuracy']:.4f}",
        f"deckexact {metrics['deckbuilding_exact_accuracy']:.4f}",
    ]
    if "deckbuild_rollout_overlap" in metrics:
        parts.extend(
            [
                f"rollout_overlap {metrics['deckbuild_rollout_overlap']:.4f}",
                f"rollout_exact {metrics['deckbuild_rollout_exact']:.4f}",
            ]
        )
    parts.extend(
        [
            f"mrr {metrics['mrr']:.4f}",
            f"nontriv {metrics['nontrivial_top1']:.4f}",
            f"{metrics['seconds']:.1f}s",
        ]
    )
    return " | ".join(parts)


def save_top1_plot(history: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] + 1 for row in history]
    train_top1 = [row["train"]["top1"] for row in history]
    val_top1 = [row["val"]["top1"] for row in history]

    TOP1_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_top1, marker="o", label="train draft top-1")
    plt.plot(epochs, val_top1, marker="o", label="validation draft top-1")
    plt.xlabel("Epoch")
    plt.ylabel("Draft top-1 accuracy")
    plt.title("Draft Top-1 Accuracy Over Epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(TOP1_PLOT, dpi=150)
    plt.close()
    print(f"saved top-1 graph to {TOP1_PLOT}")


def main() -> int:
    epochs = 1 if QUICK else EPOCHS
    batch_size = min(BATCH_SIZE, 1024) if QUICK else BATCH_SIZE
    limit_chunks = LIMIT_CHUNKS or QUICK_CHUNKS if QUICK else LIMIT_CHUNKS
    max_train_batches = MAX_STEPS_PER_EPOCH or QUICK_TRAIN_BATCHES if QUICK else MAX_STEPS_PER_EPOCH
    max_val_batches = MAX_VAL_BATCHES or QUICK_VAL_BATCHES if QUICK else MAX_VAL_BATCHES

    if NUM_THREADS > 0:
        torch.set_num_threads(NUM_THREADS)
    torch.set_float32_matmul_precision("high")
    seed_everything()

    device = torch.device(DEVICE if DEVICE != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(USE_AMP and device.type == "cuda")
    metadata = load_json(DATA_DIR / "metadata.json")
    card_names = metadata["card_names"]
    cards = len(card_names)
    basic_mask = basic_land_mask(card_names)
    basic_land_indices = np.flatnonzero(basic_mask).astype(int).tolist()
    chunks = draft_chunk_paths(DATA_DIR)[:limit_chunks]
    if not chunks:
        raise FileNotFoundError(f"No draft_xy_chunk_*.npz files found in {DATA_DIR}")
    train_build = should_train_build(chunks)
    if train_build and not basic_land_indices:
        raise ValueError("Build-phase training requires basic lands to be present in metadata card_names.")
    if ROLLOUT_TRAIN_START_EPOCH < 0:
        raise ValueError("ROLLOUT_TRAIN_START_EPOCH must be non-negative.")
    if ROLLOUT_TRAIN_DECKS_PER_CHUNK is not None and ROLLOUT_TRAIN_DECKS_PER_CHUNK <= 0:
        raise ValueError("ROLLOUT_TRAIN_DECKS_PER_CHUNK must be positive or None.")
    if ROLLOUT_TRAIN_MAX_STEPS is not None and ROLLOUT_TRAIN_MAX_STEPS <= 0:
        raise ValueError("ROLLOUT_TRAIN_MAX_STEPS must be positive or None.")
    if ROLLOUT_TRAIN_BATCH_DECKS <= 0:
        raise ValueError("ROLLOUT_TRAIN_BATCH_DECKS must be positive.")

    train_rows, val_rows, train_batches, val_batches = split_summary(chunks, cards, VAL_FRACTION, batch_size, train_build)
    train_batches = min(train_batches, max_train_batches) if max_train_batches is not None else train_batches
    val_batches = min(val_batches, max_val_batches) if max_val_batches is not None else val_batches

    features, feature_names, feature_match = build_card_features(card_names)
    model, model_kwargs = make_model(cards, features, basic_land_indices)
    model = model.to(device)
    basic_land_mask_tensor = torch.as_tensor(basic_mask, dtype=torch.bool, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(train_batches * epochs, 1), pct_start=0.08, div_factor=10.0, final_div_factor=50.0)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    start_epoch, best, history = 0, math.inf, []
    if RESUME is not None:
        start_epoch, best, history = load_checkpoint(RESUME, model, opt, scheduler, device)
    if COMPILE:
        model = torch.compile(model)

    print(f"device: {device}")
    print(f"model: {MODEL}")
    print(f"data: {DATA_DIR}")
    print(f"phases: draft={TRAIN_DRAFT_PHASE}, build={train_build}")
    print(
        "rollout train: "
        f"{bool(train_build and TRAIN_BUILD_ROLLOUT)} "
        f"(start_epoch={ROLLOUT_TRAIN_START_EPOCH + 1}, "
        f"decks_per_chunk={ROLLOUT_TRAIN_DECKS_PER_CHUNK}, "
        f"max_steps={ROLLOUT_TRAIN_MAX_STEPS})"
    )
    print(f"cards: {cards}")
    print(f"basic land actions: {len(basic_land_indices)}")
    print(f"chunks: {len(chunks)}")
    print(f"examples: train {train_rows:,}, val {val_rows:,}")
    print(f"batches/epoch: train {train_batches:,}, val {val_batches:,}")
    print(f"card metadata: matched {feature_match['matched']}, missing {feature_match['missing']}, features {len(feature_names)}")
    print(f"checkpoints: {OUTPUT_DIR}")

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train = run_epoch(model, opt, scheduler, scaler, device, chunks, cards, epoch, batch_size, VAL_FRACTION, max_train_batches, train_build, basic_land_mask_tensor)
        val = run_epoch(model, None, None, None, device, chunks, cards, epoch, batch_size, VAL_FRACTION, max_val_batches, train_build, basic_land_mask_tensor)
        if train_build and VALIDATE_BUILD_ROLLOUT:
            val.update(
                evaluate_build_rollout(
                    model=model,
                    device=device,
                    chunks=chunks,
                    cards=cards,
                    val_fraction=VAL_FRACTION,
                    basic_land_mask_tensor=basic_land_mask_tensor,
                    max_decks=ROLLOUT_VAL_DECK_LIMIT,
                )
            )
        print(format_metrics("train", train))
        print(format_metrics("val  ", val))

        history.append({"epoch": epoch, "train": train, "val": val, "lr": opt.param_groups[0]["lr"]})
        if val["loss"] < best:
            best = val["loss"]
            save_checkpoint(OUTPUT_DIR / "draft_scorer_best.pt", model, opt, scheduler, epoch, best, metadata, model_kwargs, history, feature_names, feature_match)
            print(f"saved best checkpoint with val loss {best:.4f}")
        save_checkpoint(OUTPUT_DIR / "draft_scorer_last.pt", model, opt, scheduler, epoch, best, metadata, model_kwargs, history, feature_names, feature_match)
        if SAVE_EVERY_EPOCH:
            save_checkpoint(OUTPUT_DIR / f"draft_scorer_epoch_{epoch + 1:03d}.pt", model, opt, scheduler, epoch, best, metadata, model_kwargs, history, feature_names, feature_match)

    save_top1_plot(history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
