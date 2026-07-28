import sys
from pathlib import Path

from deck_tester.evaluate_agent import evaluate_agent
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import torch
from SL.NN import ContextualDraftScorer, DraftScorer
from draft_env import draft_env

# MODEL ARCHITECTURE

    # Draft a batch of decks
    # Play all against a chosen/random combination of decks from "best drafted decks"
    # Compute Reward:
        # r = -1 if loss, 1 if win, 0 if draw
        # R = #wins/#loss
    # compute loss = PPOClipLoss()
    # back propagate and update model weights by learning rate

#setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load initial model from SL training
CHECKPOINT = ROOT / "SL" / "models" / "draft_scorer_best.pt"
checkpoint = torch.load(CHECKPOINT, map_location=device)
model_name = checkpoint.get("model_name", "contextual")
model = DraftScorer(**checkpoint["model_kwargs"])
model.load_state_dict(checkpoint["model_state"])
model.to(device)

win_rate = evaluate_agent(model,num_drafts=10,games_per_deck=25,workers=8)
print(win_rate)


