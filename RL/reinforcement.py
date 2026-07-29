#!/usr/bin/env python3
"""PPO fine-tuning for the draft-and-build agent.

The actor starts from the supervised-learning DraftScorer checkpoint. Rollouts
collect complete drafted decks; each completed deck is evaluated by Forge against
four sampled pool opponents for 20 games each, and that 80-game average win rate
is used as the terminal reward for PPO.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from RL.deck_tester.deck_tester import deck_tester
from RL.draft_env import draft_env
from SL.NN import DraftScorer


@dataclass
class PPOConfig:
    checkpoint: Path = ROOT / "SL" / "models" / "draft_scorer_best.pt"
    output_dir: Path = HERE / "models"
    resume: Path | None = None
    seed: int = 42
    device: str = "cuda"

    updates: int = 3
    batch_size: int = 16
    ppo_epochs: int = 4
    minibatch_size: int = 256
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4
    weight_decay: float = 1e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03

    value_hidden_dim: int = 256
    value_dropout: float = 0.0

    opponents_per_deck: int = 4
    games_per_opponent: int = 10
    deck_test_workers: int = 8
    best_of: int = 1
    deck_test_timeout: int = 300

    log_every: int = 1
    save_every: int = 1


@dataclass
class Transition:
    pool_counts: np.ndarray
    pack_counts: np.ndarray
    deck_counts: np.ndarray
    legal_mask: np.ndarray
    pack_number: int
    pick_number: int
    phase: int
    build_step: int
    action: int
    old_log_prob: float
    old_value: float
    reward: float = 0.0
    done: bool = False
    advantage: float = 0.0
    ret: float = 0.0


@dataclass
class RolloutBatch:
    pool_counts: torch.Tensor
    pack_counts: torch.Tensor
    deck_counts: torch.Tensor
    legal_mask: torch.Tensor
    pack_numbers: torch.Tensor
    pick_numbers: torch.Tensor
    phases: torch.Tensor
    build_steps: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ValueNetwork(nn.Module):
    """Critic over the same phase-aware state features used by DraftScorer."""

    def __init__(
        self,
        num_cards: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        max_packs: int = 3,
        max_picks: int = 14,
        max_build_steps: int = 40,
        max_pool_size: int = 45,
        max_pack_size: int = 15,
        target_deck_size: int = 40,
    ) -> None:
        super().__init__()
        self.num_cards = int(num_cards)
        self.max_packs = int(max_packs)
        self.max_picks = int(max_picks)
        self.max_build_steps = int(max_build_steps)
        self.max_pool_size = int(max_pool_size)
        self.max_pack_size = int(max_pack_size)
        self.target_deck_size = int(target_deck_size)

        input_dim = self.num_cards * 3 + 7
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        pool_counts: torch.Tensor,
        pack_counts: torch.Tensor,
        deck_counts: torch.Tensor,
        pack_numbers: torch.Tensor,
        pick_numbers: torch.Tensor,
        phases: torch.Tensor,
        build_steps: torch.Tensor,
    ) -> torch.Tensor:
        pool_counts = pool_counts.float()
        pack_counts = pack_counts.float()
        deck_counts = deck_counts.float()

        batch_size = pool_counts.shape[0]
        pack_numbers = pack_numbers.float().view(batch_size)
        pick_numbers = pick_numbers.float().view(batch_size)
        phases = phases.float().view(batch_size)
        build_steps = build_steps.float().view(batch_size)

        stage = torch.stack(
            [
                phases,
                pack_numbers / max(self.max_packs - 1, 1),
                pick_numbers / max(self.max_picks - 1, 1),
                build_steps / max(self.max_build_steps, 1),
                pool_counts.sum(1) / max(self.max_pool_size, 1),
                pack_counts.sum(1) / max(self.max_pack_size, 1),
                deck_counts.sum(1) / max(self.target_deck_size, 1),
            ],
            dim=1,
        )
        return self.net(torch.cat([pool_counts, pack_counts, deck_counts, stage], dim=1)).squeeze(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_sl_actor(checkpoint_path: Path, device: torch.device) -> tuple[DraftScorer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "contextual")
    if model_name != "draftscorer":
        raise ValueError(
            f"PPO fine-tuning requires a DraftScorer checkpoint with build-phase support; "
            f"{checkpoint_path} contains {model_name!r}."
        )

    actor = DraftScorer(**checkpoint["model_kwargs"])
    actor.load_state_dict(checkpoint["model_state"])
    actor.to(device)
    actor.eval()
    return actor, checkpoint


def make_value_network(actor: DraftScorer, cfg: PPOConfig, device: torch.device) -> ValueNetwork:
    critic = ValueNetwork(
        num_cards=actor.num_cards,
        hidden_dim=cfg.value_hidden_dim,
        dropout=cfg.value_dropout,
        max_packs=actor.max_packs,
        max_picks=actor.max_picks,
        max_build_steps=actor.max_build_steps,
        max_pool_size=actor.max_pool_size,
        max_pack_size=actor.max_pack_size,
        target_deck_size=actor.target_deck_size,
    )
    return critic.to(device)


def obs_to_tensors(obs: dict[str, Any], mask: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "pool_counts": torch.as_tensor(obs["pool_counts"], dtype=torch.float32, device=device).unsqueeze(0),
        "pack_counts": torch.as_tensor(obs["pack_counts"], dtype=torch.float32, device=device).unsqueeze(0),
        "deck_counts": torch.as_tensor(obs["deck_counts"], dtype=torch.float32, device=device).unsqueeze(0),
        "legal_mask": torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0),
        "pack_numbers": torch.tensor([obs["pack_number"]], dtype=torch.long, device=device),
        "pick_numbers": torch.tensor([obs["pick_number"]], dtype=torch.long, device=device),
        "phases": torch.tensor([obs["phase"]], dtype=torch.float32, device=device),
        "build_steps": torch.tensor([obs["build_step"]], dtype=torch.float32, device=device),
    }


def actor_logits(actor: DraftScorer, batch: dict[str, torch.Tensor] | RolloutBatch) -> torch.Tensor:
    return actor(
        batch.pool_counts if isinstance(batch, RolloutBatch) else batch["pool_counts"],
        batch.pack_counts if isinstance(batch, RolloutBatch) else batch["pack_counts"],
        batch.pack_numbers if isinstance(batch, RolloutBatch) else batch["pack_numbers"],
        batch.pick_numbers if isinstance(batch, RolloutBatch) else batch["pick_numbers"],
        deck_counts=batch.deck_counts if isinstance(batch, RolloutBatch) else batch["deck_counts"],
        phases=batch.phases if isinstance(batch, RolloutBatch) else batch["phases"],
        build_steps=batch.build_steps if isinstance(batch, RolloutBatch) else batch["build_steps"],
        legal_mask=batch.legal_mask if isinstance(batch, RolloutBatch) else batch["legal_mask"],
    )


def value_predictions(critic: ValueNetwork, batch: dict[str, torch.Tensor] | RolloutBatch) -> torch.Tensor:
    return critic(
        batch.pool_counts if isinstance(batch, RolloutBatch) else batch["pool_counts"],
        batch.pack_counts if isinstance(batch, RolloutBatch) else batch["pack_counts"],
        batch.deck_counts if isinstance(batch, RolloutBatch) else batch["deck_counts"],
        batch.pack_numbers if isinstance(batch, RolloutBatch) else batch["pack_numbers"],
        batch.pick_numbers if isinstance(batch, RolloutBatch) else batch["pick_numbers"],
        batch.phases if isinstance(batch, RolloutBatch) else batch["phases"],
        batch.build_steps if isinstance(batch, RolloutBatch) else batch["build_steps"],
    )


def sample_action(
    actor: DraftScorer,
    critic: ValueNetwork,
    obs: dict[str, Any],
    mask: np.ndarray,
    device: torch.device,
) -> tuple[int, float, float]:
    if not bool(np.any(mask)):
        raise RuntimeError("No legal actions available during rollout.")

    tensors = obs_to_tensors(obs, mask, device)
    with torch.no_grad():
        logits = actor_logits(actor, tensors)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        value = value_predictions(critic, tensors)

    return int(action.item()), float(log_prob.item()), float(value.item())


def collect_deck_batch(
    actor: DraftScorer,
    critic: ValueNetwork,
    card_names: list[str],
    cfg: PPOConfig,
    device: torch.device,
    update_index: int,
) -> tuple[list[list[Transition]], list[list[str]]]:
    actor.eval()
    critic.eval()

    episodes: list[list[Transition]] = []
    decks: list[list[str]] = []
    base_seed = cfg.seed + update_index * cfg.batch_size

    for episode_index in range(cfg.batch_size):
        env = draft_env(card_names=card_names, target_deck_size=actor.target_deck_size)
        obs, info = env.reset(seed=base_seed + episode_index)
        transitions: list[Transition] = []

        while True:
            mask = np.asarray(info["action_mask"], dtype=np.bool_)
            action, log_prob, value = sample_action(actor, critic, obs, mask, device)
            transition = Transition(
                pool_counts=obs["pool_counts"].astype(np.float32, copy=True),
                pack_counts=obs["pack_counts"].astype(np.float32, copy=True),
                deck_counts=obs["deck_counts"].astype(np.float32, copy=True),
                legal_mask=mask.copy(),
                pack_number=int(obs["pack_number"]),
                pick_number=int(obs["pick_number"]),
                phase=int(obs["phase"]),
                build_step=int(obs["build_step"]),
                action=action,
                old_log_prob=log_prob,
                old_value=value,
            )

            obs, _, terminated, truncated, info = env.step(action)
            transition.done = bool(terminated)
            transitions.append(transition)

            if truncated:
                raise RuntimeError("Draft environment truncated before producing a completed deck.")
            if terminated:
                decks.append(info["completed_deck_nametext"].splitlines())
                break

        episodes.append(transitions)

    return episodes, decks


def evaluate_decks(
    decks: list[list[str]],
    cfg: PPOConfig,
    update_index: int,
    tester: deck_tester | None = None,
    return_details: bool = False,
):
    tester = tester or deck_tester()
    result = tester.test_batch_against_pool(
        decks,
        opponents_per_deck=cfg.opponents_per_deck,
        games_per_opponent=cfg.games_per_opponent,
        best_of=cfg.best_of,
        timeout=cfg.deck_test_timeout,
        seed=cfg.seed + update_index * 100_000,
        workers=cfg.deck_test_workers,
        return_details=return_details,
    )
    if return_details:
        win_rate_percentages = result["win_rates"]
        return [win_rate / 100.0 for win_rate in win_rate_percentages], result

    win_rate_percentages = result
    return [win_rate / 100.0 for win_rate in win_rate_percentages]


def attach_terminal_rewards(episodes: list[list[Transition]], rewards: list[float]) -> None:
    if len(episodes) != len(rewards):
        raise ValueError(f"Expected {len(episodes)} rewards, got {len(rewards)}.")
    for episode, reward in zip(episodes, rewards):
        if not episode:
            raise ValueError("Cannot attach reward to an empty episode.")
        episode[-1].reward = float(reward)
        episode[-1].done = True


def compute_gae(episodes: list[list[Transition]], gamma: float, gae_lambda: float) -> None:
    for episode in episodes:
        next_value = 0.0
        gae = 0.0
        for transition in reversed(episode):
            next_nonterminal = 0.0 if transition.done else 1.0
            delta = transition.reward + gamma * next_value * next_nonterminal - transition.old_value
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            transition.advantage = gae
            transition.ret = gae + transition.old_value
            next_value = transition.old_value


def stack_rollout(episodes: list[list[Transition]], device: torch.device) -> RolloutBatch:
    transitions = [transition for episode in episodes for transition in episode]
    if not transitions:
        raise ValueError("Cannot build a rollout batch from no transitions.")

    advantages = torch.as_tensor([t.advantage for t in transitions], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

    return RolloutBatch(
        pool_counts=torch.as_tensor(np.stack([t.pool_counts for t in transitions]), dtype=torch.float32, device=device),
        pack_counts=torch.as_tensor(np.stack([t.pack_counts for t in transitions]), dtype=torch.float32, device=device),
        deck_counts=torch.as_tensor(np.stack([t.deck_counts for t in transitions]), dtype=torch.float32, device=device),
        legal_mask=torch.as_tensor(np.stack([t.legal_mask for t in transitions]), dtype=torch.bool, device=device),
        pack_numbers=torch.as_tensor([t.pack_number for t in transitions], dtype=torch.long, device=device),
        pick_numbers=torch.as_tensor([t.pick_number for t in transitions], dtype=torch.long, device=device),
        phases=torch.as_tensor([t.phase for t in transitions], dtype=torch.float32, device=device),
        build_steps=torch.as_tensor([t.build_step for t in transitions], dtype=torch.float32, device=device),
        actions=torch.as_tensor([t.action for t in transitions], dtype=torch.long, device=device),
        old_log_probs=torch.as_tensor([t.old_log_prob for t in transitions], dtype=torch.float32, device=device),
        old_values=torch.as_tensor([t.old_value for t in transitions], dtype=torch.float32, device=device),
        advantages=advantages,
        returns=torch.as_tensor([t.ret for t in transitions], dtype=torch.float32, device=device),
    )


def minibatch(batch: RolloutBatch, indices: torch.Tensor) -> RolloutBatch:
    return RolloutBatch(
        pool_counts=batch.pool_counts[indices],
        pack_counts=batch.pack_counts[indices],
        deck_counts=batch.deck_counts[indices],
        legal_mask=batch.legal_mask[indices],
        pack_numbers=batch.pack_numbers[indices],
        pick_numbers=batch.pick_numbers[indices],
        phases=batch.phases[indices],
        build_steps=batch.build_steps[indices],
        actions=batch.actions[indices],
        old_log_probs=batch.old_log_probs[indices],
        old_values=batch.old_values[indices],
        advantages=batch.advantages[indices],
        returns=batch.returns[indices],
    )


def explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    target_var = torch.var(targets)
    if float(target_var.item()) < 1e-8:
        return 0.0
    return float((1.0 - torch.var(targets - predictions) / target_var).item())


def ppo_update(
    actor: DraftScorer,
    critic: ValueNetwork,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    cfg: PPOConfig,
) -> dict[str, float]:
    actor.eval()
    critic.train()
    n = rollout.actions.shape[0]
    minibatch_size = min(cfg.minibatch_size, n)
    params = list(actor.parameters()) + list(critic.parameters())

    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "updates": 0.0,
    }

    for _ in range(cfg.ppo_epochs):
        permutation = torch.randperm(n, device=rollout.actions.device)
        stop_early = False

        for start in range(0, n, minibatch_size):
            batch_indices = permutation[start : start + minibatch_size]
            mb = minibatch(rollout, batch_indices)

            logits = actor_logits(actor, mb)
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(mb.actions)
            entropy = distribution.entropy().mean()

            log_ratio = new_log_probs - mb.old_log_probs
            ratio = log_ratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()

            unclipped_policy = -mb.advantages * ratio
            clipped_policy = -mb.advantages * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            policy_loss = torch.max(unclipped_policy, clipped_policy).mean()

            values = value_predictions(critic, mb)
            value_loss_unclipped = (values - mb.returns).pow(2)
            clipped_values = mb.old_values + torch.clamp(values - mb.old_values, -cfg.clip_coef, cfg.clip_coef)
            value_loss_clipped = (clipped_values - mb.returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()

            metrics["policy_loss"] += float(policy_loss.detach().cpu())
            metrics["value_loss"] += float(value_loss.detach().cpu())
            metrics["entropy"] += float(entropy.detach().cpu())
            metrics["approx_kl"] += float(approx_kl.detach().cpu())
            metrics["clip_fraction"] += float(clip_fraction.detach().cpu())
            metrics["updates"] += 1.0

            if cfg.target_kl is not None and float(approx_kl.detach().cpu()) > cfg.target_kl:
                stop_early = True
                break

        if stop_early:
            break

    denom = max(metrics.pop("updates"), 1.0)
    metrics = {key: value / denom for key, value in metrics.items()}
    with torch.no_grad():
        values = value_predictions(critic, rollout)
        metrics["explained_variance"] = explained_variance(values, rollout.returns)
    return metrics


def save_checkpoint(
    path: Path,
    actor: DraftScorer,
    critic: ValueNetwork,
    optimizer: torch.optim.Optimizer,
    cfg: PPOConfig,
    update_index: int,
    best_reward: float,
    sl_checkpoint: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_cfg = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(cfg).items()
    }
    torch.save(
        {
            "update": update_index,
            "best_reward": best_reward,
            "actor_state": actor.state_dict(),
            "critic_state": critic.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": serializable_cfg,
            "history": history,
            "card_names": sl_checkpoint["card_names"],
            "model_name": "draftscorer",
            "model_kwargs": sl_checkpoint["model_kwargs"],
            "sl_checkpoint": str(cfg.checkpoint),
        },
        path,
    )


def load_rl_checkpoint(
    path: Path,
    actor: DraftScorer,
    critic: ValueNetwork,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]]]:
    checkpoint = torch.load(path, map_location=device)
    actor.load_state_dict(checkpoint["actor_state"])
    critic.load_state_dict(checkpoint["critic_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return (
        int(checkpoint.get("update", 0)) + 1,
        float(checkpoint.get("best_reward", -math.inf)),
        checkpoint.get("history", []),
    )


def format_update(update_index: int, rewards: list[float], timing: dict[str, float], metrics: dict[str, float]) -> str:
    return (
        f"update {update_index:04d} | reward mean {np.mean(rewards):.3f} "
        f"min {np.min(rewards):.3f} max {np.max(rewards):.3f} | "
        f"policy {metrics['policy_loss']:.4f} value {metrics['value_loss']:.4f} "
        f"entropy {metrics['entropy']:.3f} kl {metrics['approx_kl']:.4f} "
        f"clip {metrics['clip_fraction']:.3f} ev {metrics['explained_variance']:.3f} | "
        f"collect {timing['collect']:.1f}s eval {timing['eval']:.1f}s train {timing['train']:.1f}s"
    )


def train(cfg: PPOConfig) -> list[dict[str, Any]]:
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(cfg.device)

    actor, sl_checkpoint = load_sl_actor(cfg.checkpoint, device)
    critic = make_value_network(actor, cfg, device)
    optimizer = torch.optim.AdamW(
        [
            {"params": actor.parameters(), "lr": cfg.actor_lr},
            {"params": critic.parameters(), "lr": cfg.critic_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    start_update = 0
    best_reward = -math.inf
    history: list[dict[str, Any]] = []
    if cfg.resume is not None:
        start_update, best_reward, history = load_rl_checkpoint(cfg.resume, actor, critic, optimizer, device)

    card_names = sl_checkpoint["card_names"]
    tester = deck_tester()

    print(f"device: {device}")
    print(f"actor checkpoint: {cfg.checkpoint}")
    print(f"deck batch size: {cfg.batch_size}")
    print(
        "deck reward: "
        f"{cfg.opponents_per_deck} opponents x {cfg.games_per_opponent} games "
        f"= {cfg.opponents_per_deck * cfg.games_per_opponent} games/deck, "
        f"{cfg.deck_test_workers} workers"
    )
    print(f"checkpoints: {cfg.output_dir}")

    for update_index in range(start_update, cfg.updates):
        started = time.perf_counter()
        episodes, decks = collect_deck_batch(actor, critic, card_names, cfg, device, update_index)
        collected = time.perf_counter()

        rewards, reward_details = evaluate_decks(decks, cfg, update_index, tester, return_details=True)
        evaluated = time.perf_counter()

        attach_terminal_rewards(episodes, rewards)
        compute_gae(episodes, cfg.gamma, cfg.gae_lambda)
        rollout = stack_rollout(episodes, device)
        metrics = ppo_update(actor, critic, optimizer, rollout, cfg)
        trained = time.perf_counter()

        timing = {
            "collect": collected - started,
            "eval": evaluated - collected,
            "train": trained - evaluated,
        }
        row = {
            "update": update_index,
            "rewards": rewards,
            "mean_reward": float(np.mean(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "reward_games_per_deck": cfg.opponents_per_deck * cfg.games_per_opponent,
            "reward_resolution": 1.0 / (cfg.opponents_per_deck * cfg.games_per_opponent),
            "reward_details": reward_details,
            "transitions": int(rollout.actions.shape[0]),
            "timing": timing,
            "metrics": metrics,
        }
        history.append(row)

        if cfg.log_every > 0 and (update_index + 1) % cfg.log_every == 0:
            print(format_update(update_index + 1, rewards, timing, metrics))

        if row["mean_reward"] > best_reward:
            best_reward = row["mean_reward"]
            save_checkpoint(
                cfg.output_dir / "ppo_best.pt",
                actor,
                critic,
                optimizer,
                cfg,
                update_index,
                best_reward,
                sl_checkpoint,
                history,
            )

        if cfg.save_every > 0 and (update_index + 1) % cfg.save_every == 0:
            save_checkpoint(
                cfg.output_dir / "ppo_last.pt",
                actor,
                critic,
                optimizer,
                cfg,
                update_index,
                best_reward,
                sl_checkpoint,
                history,
            )
            with (cfg.output_dir / "ppo_history.json").open("w", encoding="utf-8") as file:
                json.dump(history, file, indent=2)

    return history


def parse_args() -> PPOConfig:
    defaults = PPOConfig()
    parser = argparse.ArgumentParser(description="PPO fine-tune the MTG draft agent.")
    parser.add_argument("--checkpoint", type=Path, default=defaults.checkpoint)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--resume", type=Path, default=defaults.resume)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)

    parser.add_argument("--updates", type=int, default=defaults.updates)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--ppo-epochs", type=int, default=defaults.ppo_epochs)
    parser.add_argument("--minibatch-size", type=int, default=defaults.minibatch_size)
    parser.add_argument("--actor-lr", type=float, default=defaults.actor_lr)
    parser.add_argument("--critic-lr", type=float, default=defaults.critic_lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--gae-lambda", type=float, default=defaults.gae_lambda)
    parser.add_argument("--clip-coef", type=float, default=defaults.clip_coef)
    parser.add_argument("--value-coef", type=float, default=defaults.value_coef)
    parser.add_argument("--entropy-coef", type=float, default=defaults.entropy_coef)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--target-kl", type=float, default=defaults.target_kl)

    parser.add_argument("--value-hidden-dim", type=int, default=defaults.value_hidden_dim)
    parser.add_argument("--value-dropout", type=float, default=defaults.value_dropout)

    parser.add_argument("--opponents-per-deck", type=int, default=defaults.opponents_per_deck)
    parser.add_argument("--games-per-opponent", type=int, default=defaults.games_per_opponent)
    parser.add_argument("--deck-test-workers", type=int, default=defaults.deck_test_workers)
    parser.add_argument("--best-of", type=int, default=defaults.best_of)
    parser.add_argument("--deck-test-timeout", type=int, default=defaults.deck_test_timeout)

    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--save-every", type=int, default=defaults.save_every)

    args = parser.parse_args()
    return PPOConfig(**vars(args))


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
