from pathlib import Path

import torch

from MTG_Draft_Agent.SL.NN import ContextualDraftScorer, DraftScorer


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "draft_scorer_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT, map_location=device)
card_names = checkpoint["card_names"]
print(card_names)
card_to_index = {name: i for i, name in enumerate(card_names)}

model_name = checkpoint.get("model_name", "contextual")
model_kwargs = checkpoint["model_kwargs"]

if model_name == "draftscorer":
    model = DraftScorer(**model_kwargs)
else:
    model = ContextualDraftScorer(**model_kwargs)

model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()


def recommend_pick(pool_cards, pack_cards, pack_number, pick_number, top_k=5):
    pool_counts = torch.zeros(1, len(card_names), device=device)
    pack_counts = torch.zeros(1, len(card_names), device=device)

    for card in pool_cards:
        pool_counts[0, card_to_index[card]] += 1

    for card in pack_cards:
        pack_counts[0, card_to_index[card]] += 1

    pack_number = torch.tensor([pack_number], device=device)
    pick_number = torch.tensor([pick_number], device=device)

    with torch.no_grad():
        logits = model(pool_counts, pack_counts, pack_number, pick_number)
        probs = torch.softmax(logits, dim=1)
        values, indices = probs.topk(top_k, dim=1)

    return [
        (card_names[i], float(score))
        for i, score in zip(indices[0].tolist(), values[0].tolist())
    ]


pack = [
    "Planar Engineering",
    "Thunderdrum Soloist",
    "Eternal Student",
    "Landscape Painter",
    "Expressive Firedancer",
    "Stone Docent",
    "Adventurous Eater",
    "Pull from the Grave",
    "Sneering Shadewriter",
    "Fields of Strife",
]

pool = [
    "Tome Blast",
    "Snarl Song",
]

picks = recommend_pick(
    pool_cards=pool,
    pack_cards=pack,
    pack_number=0,
    pick_number=2,
)

for card, score in picks:
    print(f"{card}: {score:.3f}")