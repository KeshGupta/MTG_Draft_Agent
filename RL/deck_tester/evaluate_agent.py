import sys
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from RL.deck_tester.deck_tester import deck_tester
from RL.draft_env import draft_env

BASIC_LAND_NAMES = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}
CARD_METADATA_PATHS = (
    ROOT / "SOS_cards.json",
    ROOT / "SL" / "data" / "raw" / "SOS_extra_cards.json",
)


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


def draft_deck(agent, env=None, seed=None, device=None, card_names=None, target_deck_size=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(agent, torch.nn.Module):
        agent.to(device)
        agent.eval()

    if env is None:
        env_kwargs = {}
        if card_names is not None:
            env_kwargs["card_names"] = card_names
        if target_deck_size is not None:
            env_kwargs["target_deck_size"] = target_deck_size
        env = draft_env(**env_kwargs)

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


def _parse_deck_lines(deck):
    counts = Counter()
    for line in deck:
        text = str(line).strip()
        if not text:
            continue
        parts = text.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            counts[parts[1]] += int(parts[0])
        else:
            counts[text] += 1
    return counts


@lru_cache(maxsize=1)
def _card_metadata():
    records = {}
    for path in CARD_METADATA_PATHS:
        if not path.exists():
            continue
        for card in json.loads(path.read_text(encoding="utf-8")):
            name = card.get("name")
            if not name:
                continue
            records[name] = card
            for part in name.split(" // "):
                records.setdefault(part, card)
    return records


def _deck_summary(deck):
    counts = _parse_deck_lines(deck)
    records = _card_metadata()
    basic_land_counts = {
        name: counts[name]
        for name in sorted(BASIC_LAND_NAMES)
        if counts[name] > 0
    }
    lands = 0
    nonland_colors = Counter()

    for name, count in counts.items():
        record = records.get(name, {})
        is_land = name in BASIC_LAND_NAMES or "Land" in (record.get("type_line") or "")
        if is_land:
            lands += count
            continue
        for color in record.get("colors") or []:
            nonland_colors[color] += count

    total = sum(counts.values())
    return {
        "total_cards": total,
        "unique_cards": len(counts),
        "lands": lands,
        "nonlands": total - lands,
        "basic_lands": basic_land_counts,
        "nonland_colors": dict(sorted(nonland_colors.items())),
    }


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
    card_names=None,
    target_deck_size=None,
    fixed_opponents=False,
    opponents_per_deck=4,
    games_per_opponent=None,
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
        decks.append(
            draft_deck(
                agent,
                env=env,
                seed=draft_seed,
                device=device,
                card_names=card_names,
                target_deck_size=target_deck_size,
            )
        )

    tester = tester or deck_tester()
    tester_details = None
    if fixed_opponents:
        if opponents_per_deck <= 0:
            raise ValueError("opponents_per_deck must be positive.")
        if games_per_opponent is None:
            games_per_opponent = max(1, math.ceil(games_per_deck / opponents_per_deck))
        tester_result = tester.test_batch_against_pool(
            decks,
            opponents_per_deck=opponents_per_deck,
            games_per_opponent=games_per_opponent,
            best_of=best_of,
            timeout=timeout,
            seed=seed,
            workers=workers,
            return_details=return_details,
        )
        deck_win_rates = tester_result["win_rates"] if return_details else tester_result
        tester_details = tester_result if return_details else None
        total_games = num_drafts * opponents_per_deck * games_per_opponent
    else:
        tester_result = tester.test_batch(
            decks,
            num_games=games_per_deck,
            best_of=best_of,
            timeout=timeout,
            seed=seed,
            workers=workers,
            return_details=return_details,
        )
        deck_win_rates = tester_result["win_rates"] if return_details else tester_result
        tester_details = tester_result if return_details else None
        total_games = num_drafts * games_per_deck

    aggregate_win_rate = sum(deck_win_rates) / len(deck_win_rates) / 100.0

    if return_details:
        return {
            "win_rate": aggregate_win_rate,
            "num_drafts": num_drafts,
            "games_per_deck": games_per_deck,
            "total_games": total_games,
            "pool_dir": str(tester.pool_dir),
            "fixed_opponents": fixed_opponents,
            "deck_win_rates": [win_rate / 100.0 for win_rate in deck_win_rates],
            "deck_summaries": [_deck_summary(deck) for deck in decks],
            "tester": tester_details,
            "decks": decks,
        }
    return aggregate_win_rate


__all__ = ["draft_deck", "evaluate_agent"]
