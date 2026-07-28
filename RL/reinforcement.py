
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from SL.NN import ContextualDraftScorer, DraftScorer
from RL.draft_env import draft_env

# MODEL ARCHITECHTURE

    # Draft a batch of decks
    # Play all against a chosen/random combination of decks from "best drafted decks"
    # Compute Reward:
        # r = -1 if loss, 1 if win, 0 if draw   
        # R = #wins/#loss
    # compute loss = PPOClipLoss()
    # back propagate and update model weights by learning rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT   = ROOT / "models" / "draft_scorer_best.pt"
RATINGS_CSV  = ROOT / "data" / "card-ratings-2026-07-14 (1).csv"
CARDS_JSON   = ROOT / "data" / "SOS_cards.json"

checkpoint = torch.load(CHECKPOINT, map_location=device)

card_names = checkpoint["card_names"]
card_to_index = {name: i for i, name in enumerate(card_names)}
model_kwargs = checkpoint["model_kwargs"]

model = DraftScorer(**model_kwargs)
model.load_state_dict(checkpoint["model_state"])
model.to(device)

