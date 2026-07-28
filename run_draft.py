import sys
from pathlib import Path

# --- make project root importable no matter where Python is launched from ---
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from SL.NN import ContextualDraftScorer, DraftScorer
from RL.draft_env import draft_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FULL_DECK_SIZE = 40

CHECKPOINT   = ROOT / "SL" / "models" / "draft_scorer_best.pt"

ckpt = torch.load(CHECKPOINT, map_location=device)
card_names = ckpt["card_names"]                    # single source of truth

model_name = ckpt.get("model_name", "contextual")
model = (DraftScorer if model_name == "draftscorer" else ContextualDraftScorer)(**ckpt["model_kwargs"])
model.load_state_dict(ckpt["model_state"])
if model_name == "draftscorer":
    model.target_deck_size = max(FULL_DECK_SIZE, int(getattr(model, "target_deck_size", FULL_DECK_SIZE)))
model.to(device).eval()


def model_policy(obs, mask):
    legal = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    if not bool(legal.any()):
        raise RuntimeError("No legal actions available.")

    if model_name == "draftscorer":
        pool = torch.as_tensor(obs["pool_counts"], dtype=torch.float32, device=device).unsqueeze(0)
        pack = torch.as_tensor(obs["pack_counts"], dtype=torch.float32, device=device).unsqueeze(0)
        deck = torch.as_tensor(obs["deck_counts"], dtype=torch.float32, device=device).unsqueeze(0)
        pno = torch.tensor([obs["pack_number"]], device=device)
        kno = torch.tensor([obs["pick_number"]], device=device)
        phase = torch.tensor([obs["phase"]], dtype=torch.float32, device=device)
        build_step = torch.tensor([obs["build_step"]], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(
                pool,
                pack,
                pno,
                kno,
                deck_counts=deck,
                phases=phase,
                build_steps=build_step,
                legal_mask=legal,
            )
        return int(logits.argmax(1).item())

    if obs["phase"] == 1:
        return int(torch.nonzero(legal[0], as_tuple=False)[0].item())

    pool = torch.as_tensor(obs["pool_counts"], dtype=torch.float32, device=device).unsqueeze(0)
    pack = torch.as_tensor(obs["pack_counts"], dtype=torch.float32, device=device).unsqueeze(0)
    pno = torch.tensor([obs["pack_number"]], device=device)
    kno = torch.tensor([obs["pick_number"]], device=device)
    with torch.no_grad():
        logits = model(pool, pack, pno, kno)
    return int(logits.argmax(1).item())


env = draft_env(card_names=card_names)


def run_one_draft(env, policy):
    obs, info = env.reset()
    mask = info["action_mask"]
    terminated = False
    while not terminated:
        action = policy(obs, mask)
        obs, reward, terminated, _, info = env.step(action)
        mask = info.get("action_mask")
    return info


if __name__ == "__main__":
    result = run_one_draft(env, model_policy)
    print(result["completed_deck_nametext"])
