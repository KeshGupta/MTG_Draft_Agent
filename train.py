#!/usr/bin/env python3
"""Train a draft-pick model from cleaned 17Lands-style pick chunks.

The training target is the human pick among the cards currently in the pack.
Rows are streamed from compressed chunks so the full 4.5M-row dataset never has
to be concatenated in memory.
"""

from __future__ import annotations

import argparse
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


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "cleaned"
DEFAULT_CARD_DATA = PROJECT_DIR / "data" / "raw" / "SOS_cards.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "models"


class MetricAccumulator:
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.samples = 0
        self.correct = 0
        self.top3_correct = 0
        self.mrr_sum = 0.0
        self.nontrivial_samples = 0
        self.nontrivial_correct = 0

    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pack_mask: torch.Tensor,
    ) -> None:
        batch_size = targets.numel()
        self.loss_sum += float(loss.detach().cpu()) * batch_size
        self.samples += batch_size

        predictions = logits.argmax(dim=1)
        self.correct += int((predictions == targets).sum().detach().cpu())

        topk = logits.topk(k=min(3, logits.shape[1]), dim=1).indices
        self.top3_correct += int((topk == targets.unsqueeze(1)).any(dim=1).sum().detach().cpu())

        target_logits = logits.gather(1, targets.unsqueeze(1))
        ranks = (logits > target_logits).sum(dim=1).float() + 1.0
        self.mrr_sum += float((1.0 / ranks).sum().detach().cpu())

        legal_counts = pack_mask.sum(dim=1)
        nontrivial = legal_counts > 1
        if bool(nontrivial.any()):
            self.nontrivial_samples += int(nontrivial.sum().detach().cpu())
            self.nontrivial_correct += int((predictions[nontrivial] == targets[nontrivial]).sum().detach().cpu())

    def as_dict(self) -> dict[str, float]:
        samples = max(self.samples, 1)
        nontrivial_samples = max(self.nontrivial_samples, 1)
        return {
            "loss": self.loss_sum / samples,
            "top1": self.correct / samples,
            "top3": self.top3_correct / samples,
            "mrr": self.mrr_sum / samples,
            "nontrivial_top1": self.nontrivial_correct / nontrivial_samples,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MTG draft pick neural network.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Folder with draft_xy_chunk_*.npz files.")
    parser.add_argument("--card-data", type=Path, default=DEFAULT_CARD_DATA, help="Optional Scryfall card JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where checkpoints are written.")
    parser.add_argument("--model", choices=["contextual", "draftscorer"], default="contextual")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--synergy-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device string.")
    parser.add_argument("--num-threads", type=int, default=0, help="CPU torch thread count. 0 keeps PyTorch default.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a checkpoint.")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Use only the first N chunks.")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="Cap train batches per epoch.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Cap validation batches.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--no-card-features", action="store_true", help="Do not use Scryfall metadata features.")
    parser.add_argument("--no-choice-weighting", action="store_false", dest="choice_weighting")
    parser.set_defaults(choice_weighting=True)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: one short epoch on a few chunks.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_metadata(data_dir: Path) -> dict[str, Any]:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if "card_names" not in metadata:
        raise ValueError("metadata.json must contain card_names.")
    return metadata


def find_chunks(data_dir: Path, limit_chunks: int | None) -> list[Path]:
    chunks = sorted(data_dir.glob("draft_xy_chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"No draft_xy_chunk_*.npz files found in {data_dir}")
    if limit_chunks is not None:
        chunks = chunks[:limit_chunks]
    return chunks


def parse_number(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_card_lookup(card_data_path: Path) -> dict[str, dict[str, Any]]:
    with card_data_path.open("r", encoding="utf-8") as file:
        cards = json.load(file)

    lookup: dict[str, dict[str, Any]] = {}
    for card in cards:
        name = card.get("name")
        if not name:
            continue
        lookup.setdefault(name, card)
        for part in name.split(" // "):
            lookup.setdefault(part, card)
    return lookup


def build_card_features(
    card_names: list[str],
    card_data_path: Path,
    use_card_features: bool,
) -> tuple[torch.Tensor, list[str], dict[str, int]]:
    if not use_card_features or not card_data_path.exists():
        return torch.empty(len(card_names), 0), [], {"matched": 0, "missing": len(card_names)}

    lookup = build_card_lookup(card_data_path)
    colors = ["W", "U", "B", "R", "G"]
    type_flags = ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Land", "Planeswalker", "Battle"]
    rarity_flags = ["common", "uncommon", "rare", "mythic"]
    keyword_flags = [
        "Flying",
        "Trample",
        "Vigilance",
        "Haste",
        "Reach",
        "Deathtouch",
        "Lifelink",
        "Menace",
        "Ward",
        "Prowess",
        "Flash",
        "First strike",
        "Double strike",
        "Converge",
    ]

    feature_names = (
        [f"color_{color}" for color in colors]
        + ["is_colorless", "is_multicolor"]
        + [f"type_{type_name.lower()}" for type_name in type_flags]
        + [f"rarity_{rarity}" for rarity in rarity_flags]
        + [f"keyword_{keyword.lower().replace(' ', '_')}" for keyword in keyword_flags]
        + ["cmc_scaled", "power_scaled", "toughness_scaled", "oracle_len_scaled", "keyword_count_scaled"]
        + ["metadata_missing"]
    )

    rows: list[list[float]] = []
    matched = 0
    for card_name in card_names:
        card = lookup.get(card_name)
        missing = card is None
        if missing:
            card = {}
        else:
            matched += 1

        card_colors = set(card.get("colors") or [])
        type_line = card.get("type_line") or ""
        rarity = card.get("rarity") or ""
        keywords = set(card.get("keywords") or [])
        oracle_text = card.get("oracle_text") or ""

        row: list[float] = []
        row.extend(1.0 if color in card_colors else 0.0 for color in colors)
        row.append(1.0 if not card_colors and not missing else 0.0)
        row.append(1.0 if len(card_colors) > 1 else 0.0)
        row.extend(1.0 if type_name in type_line else 0.0 for type_name in type_flags)
        row.extend(1.0 if rarity == rarity_name else 0.0 for rarity_name in rarity_flags)
        row.extend(1.0 if keyword in keywords else 0.0 for keyword in keyword_flags)
        row.append(min(parse_number(card.get("cmc")), 12.0) / 12.0)
        row.append(min(parse_number(card.get("power")), 12.0) / 12.0)
        row.append(min(parse_number(card.get("toughness")), 12.0) / 12.0)
        row.append(min(len(oracle_text), 500) / 500.0)
        row.append(min(len(keywords), 8) / 8.0)
        row.append(1.0 if missing else 0.0)
        rows.append(row)

    features = torch.tensor(rows, dtype=torch.float32)
    return features, feature_names, {"matched": matched, "missing": len(card_names) - matched}


def chunk_row_counts(chunks: list[Path]) -> list[int]:
    counts: list[int] = []
    for path in chunks:
        with np.load(path) as data:
            counts.append(int(data["y"].shape[0]))
    return counts


def validation_mask(row_count: int, chunk_index: int, val_fraction: float, seed: int) -> np.ndarray:
    if val_fraction <= 0:
        return np.zeros(row_count, dtype=bool)
    rng = np.random.default_rng(seed + chunk_index * 1_000_003)
    return rng.random(row_count) < val_fraction


def split_summary(
    row_counts: list[int],
    val_fraction: float,
    seed: int,
    batch_size: int,
) -> tuple[int, int, int, int]:
    train_rows = 0
    val_rows = 0
    train_batches = 0
    val_batches = 0
    for chunk_index, row_count in enumerate(row_counts):
        mask = validation_mask(row_count, chunk_index, val_fraction, seed)
        chunk_val_rows = int(mask.sum())
        chunk_train_rows = row_count - chunk_val_rows
        train_rows += chunk_train_rows
        val_rows += chunk_val_rows
        train_batches += math.ceil(chunk_train_rows / batch_size) if chunk_train_rows else 0
        val_batches += math.ceil(chunk_val_rows / batch_size) if chunk_val_rows else 0
    return train_rows, val_rows, train_batches, val_batches


def iter_batches(
    chunks: list[Path],
    num_cards: int,
    batch_size: int,
    val_fraction: float,
    seed: int,
    epoch: int,
    split: str,
    max_batches: int | None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    chunk_order = list(enumerate(chunks))
    rng = np.random.default_rng(seed + epoch * 10_007 + (0 if split == "train" else 50_000))
    if split == "train":
        rng.shuffle(chunk_order)

    batches_seen = 0
    for chunk_index, path in chunk_order:
        with np.load(path) as data:
            x = data["X"]
            y = data["y"]
            val_mask = validation_mask(len(y), chunk_index, val_fraction, seed)
            indices = np.flatnonzero(~val_mask if split == "train" else val_mask)
            if split == "train":
                rng.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                if max_batches is not None and batches_seen >= max_batches:
                    return
                batch_indices = indices[start : start + batch_size]
                if len(batch_indices) == 0:
                    continue
                batch_x = np.ascontiguousarray(x[batch_indices, : 2 + 2 * num_cards])
                batch_y = np.ascontiguousarray(y[batch_indices]).astype(np.int64, copy=False)
                batches_seen += 1
                yield batch_x, batch_y


def unpack_batch(
    batch_x: np.ndarray,
    batch_y: np.ndarray,
    num_cards: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pack_numbers = torch.as_tensor(batch_x[:, 0], dtype=torch.long, device=device)
    pick_numbers = torch.as_tensor(batch_x[:, 1], dtype=torch.long, device=device)
    pack_counts = torch.as_tensor(batch_x[:, 2 : 2 + num_cards], dtype=torch.float32, device=device)
    pool_counts = torch.as_tensor(batch_x[:, 2 + num_cards : 2 + 2 * num_cards], dtype=torch.float32, device=device)
    targets = torch.as_tensor(batch_y, dtype=torch.long, device=device)
    return pool_counts, pack_counts, pack_numbers, pick_numbers, targets


def forward_model(
    model: nn.Module,
    model_name: str,
    pool_counts: torch.Tensor,
    pack_counts: torch.Tensor,
    pack_numbers: torch.Tensor,
    pick_numbers: torch.Tensor,
) -> torch.Tensor:
    if model_name == "draftscorer":
        return model(pool_counts, pack_counts > 0)
    return model(pool_counts, pack_counts, pack_numbers, pick_numbers)


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pack_mask: torch.Tensor,
    label_smoothing: float,
    choice_weighting: bool,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    if label_smoothing > 0:
        legal = pack_mask.float()
        smooth = -(log_probs * legal).sum(dim=1) / legal.sum(dim=1).clamp_min(1.0)
        per_sample = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    else:
        per_sample = nll

    if choice_weighting:
        legal_counts = pack_mask.sum(dim=1).float()
        weights = torch.log2(legal_counts.clamp_min(2.0))
        return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)
    return per_sample.mean()


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda")
        return torch.cuda.amp.autocast()
    return nullcontext()


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    model_name: str,
    chunks: list[Path],
    num_cards: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    metrics = MetricAccumulator()
    start_time = time.time()

    for step, (batch_x, batch_y) in enumerate(
        iter_batches(
            chunks=chunks,
            num_cards=num_cards,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
            epoch=epoch,
            split="train",
            max_batches=args.max_steps_per_epoch,
        ),
        start=1,
    ):
        pool_counts, pack_counts, pack_numbers, pick_numbers, targets = unpack_batch(
            batch_x=batch_x,
            batch_y=batch_y,
            num_cards=num_cards,
            device=device,
        )
        pack_mask = pack_counts > 0

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.amp):
            logits = forward_model(model, model_name, pool_counts, pack_counts, pack_numbers, pick_numbers)
            loss = masked_cross_entropy(
                logits=logits,
                targets=targets,
                pack_mask=pack_mask,
                label_smoothing=args.label_smoothing,
                choice_weighting=args.choice_weighting,
            )

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        metrics.update(loss, logits.detach(), targets, pack_mask)
        if args.log_every > 0 and step % args.log_every == 0:
            current = metrics.as_dict()
            rows_per_second = metrics.samples / max(time.time() - start_time, 1e-6)
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  step {step:5d} | loss {current['loss']:.4f} | "
                f"top1 {current['top1']:.4f} | nontriv {current['nontrivial_top1']:.4f} | "
                f"lr {lr:.2e} | {rows_per_second:,.0f} rows/s"
            )

    result = metrics.as_dict()
    result["seconds"] = time.time() - start_time
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    model_name: str,
    chunks: list[Path],
    num_cards: int,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    metrics = MetricAccumulator()
    start_time = time.time()
    for batch_x, batch_y in iter_batches(
        chunks=chunks,
        num_cards=num_cards,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        epoch=epoch,
        split="val",
        max_batches=args.max_val_batches,
    ):
        pool_counts, pack_counts, pack_numbers, pick_numbers, targets = unpack_batch(
            batch_x=batch_x,
            batch_y=batch_y,
            num_cards=num_cards,
            device=device,
        )
        pack_mask = pack_counts > 0
        logits = forward_model(model, model_name, pool_counts, pack_counts, pack_numbers, pick_numbers)
        loss = masked_cross_entropy(
            logits=logits,
            targets=targets,
            pack_mask=pack_mask,
            label_smoothing=0.0,
            choice_weighting=False,
        )
        metrics.update(loss, logits, targets, pack_mask)

    result = metrics.as_dict()
    result["seconds"] = time.time() - start_time
    return result


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def unwrap_compiled(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_val_loss: float,
    metadata: dict[str, Any],
    model_kwargs: dict[str, Any],
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    feature_names: list[str],
    feature_match: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_name": args.model,
            "model_kwargs": model_kwargs,
            "model_state": unwrap_compiled(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "args": serializable_args(args),
            "history": history,
            "card_names": metadata["card_names"],
            "feature_names": feature_names,
            "feature_match": feature_match,
        },
        path,
    )


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix} loss {metrics['loss']:.4f} | top1 {metrics['top1']:.4f} | "
        f"top3 {metrics['top3']:.4f} | mrr {metrics['mrr']:.4f} | "
        f"nontriv {metrics['nontrivial_top1']:.4f} | {metrics['seconds']:.1f}s"
    )


def make_model(
    args: argparse.Namespace,
    num_cards: int,
    card_features: torch.Tensor,
) -> tuple[nn.Module, dict[str, Any]]:
    if args.model == "draftscorer":
        kwargs = {"num_cards": num_cards, "emb_dim": args.emb_dim}
        return DraftScorer(**kwargs), kwargs

    kwargs = {
        "num_cards": num_cards,
        "card_features": card_features,
        "emb_dim": args.emb_dim,
        "hidden_dim": args.hidden_dim,
        "synergy_dim": args.synergy_dim,
        "dropout": args.dropout,
    }
    return ContextualDraftScorer(**kwargs), kwargs


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_val_loss", math.inf)), checkpoint.get("history", [])


def main() -> int:
    args = parse_args()
    if args.quick:
        args.epochs = 1
        args.limit_chunks = args.limit_chunks or 3
        args.max_steps_per_epoch = args.max_steps_per_epoch or 20
        args.max_val_batches = args.max_val_batches or 10
        args.batch_size = min(args.batch_size, 1024)

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    set_seed(args.seed)
    device = choose_device(args.device)
    args.amp = bool(args.amp and device.type == "cuda")

    metadata = load_metadata(args.data_dir)
    card_names = metadata["card_names"]
    num_cards = len(card_names)
    chunks = find_chunks(args.data_dir, args.limit_chunks)
    row_counts = chunk_row_counts(chunks)
    train_rows, val_rows, train_batches, val_batches = split_summary(
        row_counts=row_counts,
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    if args.max_steps_per_epoch is not None:
        train_batches = min(train_batches, args.max_steps_per_epoch)
    if args.max_val_batches is not None:
        val_batches = min(val_batches, args.max_val_batches)

    card_features, feature_names, feature_match = build_card_features(
        card_names=card_names,
        card_data_path=args.card_data,
        use_card_features=not args.no_card_features,
    )
    model, model_kwargs = make_model(args, num_cards, card_features)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(train_batches * args.epochs, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.08,
        div_factor=10.0,
        final_div_factor=50.0,
    )
    scaler = make_grad_scaler(enabled=args.amp)

    start_epoch = 0
    best_val_loss = math.inf
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        start_epoch, best_val_loss, history = load_checkpoint(args.resume, model, optimizer, scheduler, device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    print(f"device: {device}")
    print(f"model: {args.model}")
    print(f"cards: {num_cards}")
    print(f"chunks: {len(chunks)}")
    print(f"rows: train {train_rows:,}, val {val_rows:,}")
    print(f"batches/epoch: train {train_batches:,}, val {val_batches:,}")
    print(f"card metadata: matched {feature_match['matched']}, missing {feature_match['missing']}, features {len(feature_names)}")
    print(f"checkpoints: {args.output_dir}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_metrics = train_one_epoch(
            model=model,
            model_name=args.model,
            chunks=chunks,
            num_cards=num_cards,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            args=args,
            epoch=epoch,
        )
        print(format_metrics("train", train_metrics))

        val_metrics = evaluate(
            model=model,
            model_name=args.model,
            chunks=chunks,
            num_cards=num_cards,
            device=device,
            args=args,
            epoch=epoch,
        )
        print(format_metrics("val  ", val_metrics))

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                path=args.output_dir / "draft_scorer_best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                metadata=metadata,
                model_kwargs=model_kwargs,
                args=args,
                history=history,
                feature_names=feature_names,
                feature_match=feature_match,
            )
            print(f"saved best checkpoint with val loss {best_val_loss:.4f}")

        save_checkpoint(
            path=args.output_dir / "draft_scorer_last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            metadata=metadata,
            model_kwargs=model_kwargs,
            args=args,
            history=history,
            feature_names=feature_names,
            feature_match=feature_match,
        )
        if args.save_every_epoch:
            save_checkpoint(
                path=args.output_dir / f"draft_scorer_epoch_{epoch + 1:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                metadata=metadata,
                model_kwargs=model_kwargs,
                args=args,
                history=history,
                feature_names=feature_names,
                feature_match=feature_match,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
