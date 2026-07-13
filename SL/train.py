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

from MTG_Draft_Agent.SL.NN import ContextualDraftScorer, DraftScorer


# ---- config: edit these, then run `python train.py` -------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "cleaned"
CARD_DATA = ROOT / "data" / "raw" / "SOS_cards.json"
EXTRA_CARD_DATA = ROOT / "data" / "raw" / "SOS_extra_cards.json"
OUTPUT_DIR = ROOT / "models"
TOP1_PLOT = OUTPUT_DIR / "top1_over_epochs.png"

MODEL = "draftscorer"          # "contextual" or "draftscorer"
EPOCHS = 20
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
USE_CARD_FEATURES = True
CHOICE_WEIGHTING = True

QUICK = False                 # one short epoch on a few chunks
QUICK_CHUNKS = 3
QUICK_TRAIN_BATCHES = 20
QUICK_VAL_BATCHES = 10
# ----------------------------------------------------------------------------


class Metrics:
    def __init__(self) -> None:
        self.loss_sum = self.samples = self.correct = self.top3_correct = 0
        self.mrr_sum = self.nontrivial_samples = self.nontrivial_correct = 0

    def update(self, loss: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> None:
        n = targets.numel()
        pred = logits.argmax(1)
        target_logits = logits.gather(1, targets[:, None])
        hard = mask.sum(1) > 1

        self.loss_sum += float(loss.detach().cpu()) * n
        self.samples += n
        self.correct += int((pred == targets).sum().detach().cpu())
        self.top3_correct += int((logits.topk(min(3, logits.shape[1]), 1).indices == targets[:, None]).any(1).sum().cpu())
        self.mrr_sum += float((1.0 / ((logits > target_logits).sum(1).float() + 1.0)).sum().detach().cpu())
        self.nontrivial_samples += int(hard.sum().detach().cpu())
        self.nontrivial_correct += int((pred[hard] == targets[hard]).sum().detach().cpu()) if bool(hard.any()) else 0

    def as_dict(self) -> dict[str, float]:
        n, h = max(self.samples, 1), max(self.nontrivial_samples, 1)
        return {
            "loss": self.loss_sum / n,
            "top1": self.correct / n,
            "top3": self.top3_correct / n,
            "mrr": self.mrr_sum / n,
            "nontrivial_top1": self.nontrivial_correct / h,
        }


def config_dict() -> dict[str, Any]:
    keys = [
        "DATA_DIR", "CARD_DATA", "EXTRA_CARD_DATA", "OUTPUT_DIR", "MODEL", "EPOCHS", "BATCH_SIZE", "LR",
        "WEIGHT_DECAY", "LABEL_SMOOTHING", "VAL_FRACTION", "SEED", "EMB_DIM",
        "HIDDEN_DIM", "SYNERGY_DIM", "DROPOUT", "GRAD_CLIP", "DEVICE", "NUM_THREADS",
        "USE_AMP", "COMPILE", "RESUME", "LIMIT_CHUNKS", "MAX_STEPS_PER_EPOCH",
        "MAX_VAL_BATCHES", "LOG_EVERY", "SAVE_EVERY_EPOCH", "USE_CARD_FEATURES",
        "CHOICE_WEIGHTING", "QUICK", "QUICK_CHUNKS", "QUICK_TRAIN_BATCHES",
        "QUICK_VAL_BATCHES", "TOP1_PLOT",
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


def chunk_counts(chunks: list[Path]) -> list[int]:
    counts = []
    for path in chunks:
        with np.load(path) as data:
            counts.append(int(data["y"].shape[0]))
    return counts


def split_summary(counts: list[int], val_fraction: float, batch_size: int) -> tuple[int, int, int, int]:
    train = val = train_batches = val_batches = 0
    for i, rows in enumerate(counts):
        val_rows = int(validation_mask(rows, i, val_fraction).sum()) if val_fraction > 0 else 0
        train_rows = rows - val_rows
        train += train_rows
        val += val_rows
        train_batches += math.ceil(train_rows / batch_size) if train_rows else 0
        val_batches += math.ceil(val_rows / batch_size) if val_rows else 0
    return train, val, train_batches, val_batches


def iter_batches(
    chunks: list[Path],
    cards: int,
    batch_size: int,
    val_fraction: float,
    epoch: int,
    split: str,
    max_batches: int | None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
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

            for start in range(0, len(rows), batch_size):
                if max_batches is not None and seen >= max_batches:
                    return
                batch_rows = rows[start : start + batch_size]
                seen += 1
                yield np.ascontiguousarray(x[batch_rows, : 2 + 2 * cards]), np.ascontiguousarray(y[batch_rows]).astype(np.int64, copy=False)


def unpack(x: np.ndarray, y: np.ndarray, cards: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(x[:, 2 + cards : 2 + 2 * cards], dtype=torch.float32, device=device),
        torch.as_tensor(x[:, 2 : 2 + cards], dtype=torch.float32, device=device),
        torch.as_tensor(x[:, 0], dtype=torch.long, device=device),
        torch.as_tensor(x[:, 1], dtype=torch.long, device=device),
        torch.as_tensor(y, dtype=torch.long, device=device),
    )


def logits_for(model: nn.Module, pool: torch.Tensor, pack: torch.Tensor, pack_no: torch.Tensor, pick_no: torch.Tensor) -> torch.Tensor:
    return model(pool, pack, pack_no, pick_no)


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, smoothing: float, choice_weighting: bool) -> torch.Tensor:
    log_probs = F.log_softmax(logits, 1)
    per_row = -log_probs.gather(1, targets[:, None]).squeeze(1)
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
) -> dict[str, float]:
    training = opt is not None
    model.train(training)
    metrics, start = Metrics(), time.time()
    split = "train" if training else "val"

    for step, (x, y) in enumerate(iter_batches(chunks, cards, batch_size, val_fraction, epoch, split, max_batches), 1):
        pool, pack, pack_no, pick_no, targets = unpack(x, y, cards, device)
        mask = pack > 0
        with torch.set_grad_enabled(training), amp_context(device, USE_AMP and training):
            logits = logits_for(model, pool, pack, pack_no, pick_no)
            loss = masked_cross_entropy(logits, targets, mask, LABEL_SMOOTHING if training else 0.0, CHOICE_WEIGHTING if training else False)

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

        metrics.update(loss, logits.detach(), targets, mask)
        if training and LOG_EVERY > 0 and step % LOG_EVERY == 0:
            now = metrics.as_dict()
            speed = metrics.samples / max(time.time() - start, 1e-6)
            print(f"  step {step:5d} | loss {now['loss']:.4f} | top1 {now['top1']:.4f} | nontriv {now['nontrivial_top1']:.4f} | lr {opt.param_groups[0]['lr']:.2e} | {speed:,.0f} rows/s")

    result = metrics.as_dict()
    result["seconds"] = time.time() - start
    return result


def make_model(cards: int, features: torch.Tensor) -> tuple[nn.Module, dict[str, Any]]:
    if MODEL == "draftscorer":
        kwargs = {
            "num_cards": cards,
            "emb_dim": EMB_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
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
    return f"{prefix} loss {metrics['loss']:.4f} | top1 {metrics['top1']:.4f} | top3 {metrics['top3']:.4f} | mrr {metrics['mrr']:.4f} | nontriv {metrics['nontrivial_top1']:.4f} | {metrics['seconds']:.1f}s"


def save_top1_plot(history: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] + 1 for row in history]
    train_top1 = [row["train"]["top1"] for row in history]
    val_top1 = [row["val"]["top1"] for row in history]

    TOP1_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_top1, marker="o", label="train top-1")
    plt.plot(epochs, val_top1, marker="o", label="validation top-1")
    plt.xlabel("Epoch")
    plt.ylabel("Top-1 accuracy")
    plt.title("Top-1 Accuracy Over Epochs")
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
    chunks = sorted(DATA_DIR.glob("draft_xy_chunk_*.npz"))[:limit_chunks]

    train_rows, val_rows, train_batches, val_batches = split_summary(chunk_counts(chunks), VAL_FRACTION, batch_size)
    train_batches = min(train_batches, max_train_batches) if max_train_batches is not None else train_batches
    val_batches = min(val_batches, max_val_batches) if max_val_batches is not None else val_batches

    features, feature_names, feature_match = build_card_features(card_names)
    model, model_kwargs = make_model(cards, features)
    model = model.to(device)
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
    print(f"cards: {cards}")
    print(f"chunks: {len(chunks)}")
    print(f"rows: train {train_rows:,}, val {val_rows:,}")
    print(f"batches/epoch: train {train_batches:,}, val {val_batches:,}")
    print(f"card metadata: matched {feature_match['matched']}, missing {feature_match['missing']}, features {len(feature_names)}")
    print(f"checkpoints: {OUTPUT_DIR}")

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train = run_epoch(model, opt, scheduler, scaler, device, chunks, cards, epoch, batch_size, VAL_FRACTION, max_train_batches)
        val = run_epoch(model, None, None, None, device, chunks, cards, epoch, batch_size, VAL_FRACTION, max_val_batches)
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
