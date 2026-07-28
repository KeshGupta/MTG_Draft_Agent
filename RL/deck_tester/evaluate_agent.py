import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from RL.deck_tester.deck_tester import deck_tester
from RL.draft_env import draft_env


def _as_tensor(values, device, dtype=torch.float32):
    return torch.as_tensor(values, dtype=dtype, device=device).unsqueeze(0)


def _call_torch_agent(agent, obs, mask, device):
    pool = _as_tensor(obs["pool_counts"], device)
    pack = _as_tensor(obs["pack_counts"], device)
    deck = _as_tensor(obs["deck_counts"], device)
    legal = _as_tensor(mask, device, dtype=torch.bool)
    pack_number = torch.tensor([obs["pack_number"]], device=device)
    pick_number = torch.tensor([obs["pick_number"]], device=device)
    phase = torch.tensor([obs["phase"]], dtype=torch.float32, device=device)
    build_step = torch.tensor([obs["build_step"]], dtype=torch.float32, device=device)

    try:
        logits = agent(
            pool,
            pack,
            pack_number,
            pick_number,
            deck_counts=deck,
            phases=phase,
            build_steps=build_step,
            legal_mask=legal,
        )
    except TypeError:
        if obs["phase"] == 1:
            legal_indices = torch.nonzero(legal[0], as_tuple=False)
            return int(legal_indices[0].item())
        logits = agent(pool, pack, pack_number, pick_number)
        logits = logits.masked_fill(~legal, -1e9)

    return int(logits.argmax(dim=1).item())


def _agent_action(agent, obs, mask, device):
    if hasattr(agent, "act"):
        return int(agent.act(obs, mask))
    if hasattr(agent, "select_action"):
        return int(agent.select_action(obs, mask))
    if isinstance(agent, torch.nn.Module):
        with torch.no_grad():
            return _call_torch_agent(agent, obs, mask, device)
    if callable(agent):
        return int(agent(obs, mask))
    raise TypeError("agent must be callable, a torch.nn.Module, or expose act/select_action.")


def draft_deck(agent, env=None, seed=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(agent, torch.nn.Module):
        agent.to(device)
        agent.eval()

    env = env or draft_env()
    obs, info = env.reset(seed=seed)
    terminated = False

    while not terminated:
        mask = info["action_mask"]
        if not bool(np.any(mask)):
            raise RuntimeError("No legal actions available while drafting deck.")
        action = _agent_action(agent, obs, mask, device)
        obs, _, terminated, truncated, info = env.step(action)
        if truncated:
            raise RuntimeError("Draft environment truncated before completing a deck.")

    return info["completed_deck_nametext"].splitlines()


def evaluate_agent(
    agent,
    n_games=None,
    *,
    num_drafts=1,
    games_per_deck=None,
    env=None,
    seed=None,
    tester=None,
    best_of=1,
    timeout=300,
    workers=4,
    device=None,
    return_details=False,
):
    if games_per_deck is None:
        games_per_deck = n_games
    if games_per_deck is None:
        raise ValueError("games_per_deck is required.")
    if num_drafts <= 0:
        raise ValueError("num_drafts must be positive.")
    if games_per_deck <= 0:
        raise ValueError("games_per_deck must be positive.")

    decks = []
    for draft_index in range(num_drafts):
        draft_seed = None if seed is None else seed + draft_index
        decks.append(draft_deck(agent, env=env, seed=draft_seed, device=device))

    tester = tester or deck_tester()
    deck_win_rates = tester.test_batch(
        decks,
        num_games=games_per_deck,
        best_of=best_of,
        timeout=timeout,
        seed=seed,
        workers=workers,
    )
    aggregate_win_rate = sum(deck_win_rates) / len(deck_win_rates) / 100.0

    if return_details:
        return {
            "win_rate": aggregate_win_rate,
            "num_drafts": num_drafts,
            "games_per_deck": games_per_deck,
            "total_games": num_drafts * games_per_deck,
            "deck_win_rates": [win_rate / 100.0 for win_rate in deck_win_rates],
            "decks": decks,
        }
    return aggregate_win_rate


__all__ = ["draft_deck", "evaluate_agent"]
