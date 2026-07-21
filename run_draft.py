import sys
from pathlib import Path

# --- make project root importable no matter where Python is launched from ---
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from SL.NN import ContextualDraftScorer, DraftScorer
from RL.Deckbuilding import build_deck, load_ratings_table
from RL.draft_env import draft_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- all paths resolved from ROOT, not the working directory ---
CHECKPOINT   = ROOT / "models" / "draft_scorer_best.pt"
RATINGS_CSV  = ROOT / "data" / "card-ratings-2026-07-14 (1).csv"
CARDS_JSON   = ROOT / "data" / "SOS_cards.json"

ckpt = torch.load(CHECKPOINT, map_location=device)
card_names = ckpt["card_names"]                    # single source of truth
index_to_card = {i: n for i, n in enumerate(card_names)}

model_name = ckpt.get("model_name", "contextual")
model = (DraftScorer if model_name == "draftscorer" else ContextualDraftScorer)(**ckpt["model_kwargs"])
model.load_state_dict(ckpt["model_state"])
model.to(device).eval()


def model_policy(obs, mask):
    pool = torch.as_tensor(obs["pool_counts"], device=device).unsqueeze(0)
    pack = torch.as_tensor(obs["pack_counts"], device=device).unsqueeze(0)
    pno  = torch.tensor([obs["pack_number"]], device=device)
    kno  = torch.tensor([obs["pick_number"]], device=device)
    with torch.no_grad():
        logits = model(pool, pack, pno, kno)
    return int(logits.argmax(1))


table = load_ratings_table(RATINGS_CSV, CARDS_JSON)   # pass resolved paths in
env = draft_env(card_names=card_names)                # pass the canonical order in


def run_one_draft(env, policy):
    obs, info = env.reset()
    mask = info["action_mask"]
    terminated = False
    while not terminated:
        action = policy(obs, mask)
        obs, reward, terminated, _, info = env.step(action)
        mask = info.get("action_mask")
    return build_deck(info["final_pool"], index_to_card, table)


if __name__ == "__main__":
    deck = run_one_draft(env, model_policy)
    print(len(deck), deck)