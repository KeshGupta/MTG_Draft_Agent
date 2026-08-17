from pathlib import Path

import torch

from NN import ContextualDraftScorer, DraftScorer


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "draft_scorer_best.pt"
last_pick = False
FULL_DECK_SIZE = 40
BASIC_LAND_NAMES = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT, map_location=device)
card_names = checkpoint["card_names"]
card_to_index = {name: i for i, name in enumerate(card_names)}
basic_land_mask = torch.zeros(1, len(card_names), dtype=torch.bool, device=device)
for basic_name in BASIC_LAND_NAMES:
    if basic_name in card_to_index:
        basic_land_mask[0, card_to_index[basic_name]] = True

model_name = checkpoint.get("model_name", "contextual")
model_kwargs = checkpoint["model_kwargs"]

if model_name == "draftscorer":
    model = DraftScorer(**model_kwargs)
else:
    model = ContextualDraftScorer(**model_kwargs)

model.load_state_dict(checkpoint["model_state"])
model.to(device)
if model_name == "draftscorer":
    model.target_deck_size = max(FULL_DECK_SIZE, int(getattr(model, "target_deck_size", FULL_DECK_SIZE)))
model.eval()


def counts_from_names(cards):
    counts = torch.zeros(1, len(card_names), device=device)
    for card in cards:
        counts[0, card_to_index[card]] += 1
    return counts


def print_cards(title, counts):
    print(f"\n{title} ({int(counts.sum().item())} cards):")
    counts = counts.detach().cpu().numpy().astype(int)
    for i, count in enumerate(counts):
        if count:
            print(f"{count}x {card_names[i]}")


def choose_pick(pool_cards, pack_cards, pack_number, pick_number):
    pool_counts = counts_from_names(pool_cards)
    pack_counts = counts_from_names(pack_cards)
    pack_number = torch.tensor([pack_number], device=device)
    pick_number = torch.tensor([pick_number], device=device)

    with torch.no_grad():
        if model_name == "draftscorer":
            logits = model(
                pool_counts,
                pack_counts,
                pack_number,
                pick_number,
                deck_counts=torch.zeros_like(pool_counts),
                phases=torch.zeros(1, device=device),
                build_steps=torch.zeros(1, device=device),
                legal_mask=pack_counts > 0,
            )
        else:
            logits = model(pool_counts, pack_counts, pack_number, pick_number)

        probs = torch.softmax(logits, dim=1)
        pickable = torch.nonzero(pack_counts[0] > 0, as_tuple=False).squeeze(1)
        values = probs[0, pickable]
        order = torch.argsort(values, descending=True)

    return [
        (card_names[int(pickable[i].item())], float(values[i].item()))
        for i in order
    ]


def build_deck(final_pool_counts, target_size=FULL_DECK_SIZE):
    if model_name != "draftscorer":
        raise RuntimeError("Deckbuilding requires a draftscorer checkpoint.")
    if target_size <= 0:
        raise ValueError(f"Target deck size must be positive, got {target_size}.")

    pool_counts = final_pool_counts.clone().to(device)
    deck_counts = torch.zeros_like(pool_counts)
    pack_counts = torch.zeros_like(pool_counts)
    pack_number = torch.tensor([2], device=device)
    pick_number = torch.tensor([13], device=device)

    while int(deck_counts.sum().item()) < target_size:
        build_step = torch.tensor([int(deck_counts.sum().item())], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(
                pool_counts,
                pack_counts,
                pack_number,
                pick_number,
                deck_counts=deck_counts,
                phases=torch.ones(1, device=device),
                build_steps=build_step,
                legal_mask=(pool_counts > deck_counts) | basic_land_mask,
            )
            add_index = int(logits.argmax(1).item())

        deck_counts[0, add_index] += 1
    return deck_counts.squeeze(0)


pack = [
    "Lecturing Scornmage",
    "Chelonian Tackle",
    "Giant Growth",
    "Honorbound Page",
    "Last Gasp",
    "Burrog Barrage",
    "Follow the Lumarets",
    "Forum of Amity"
]
pool = [
    "Slumbering Trudge",
    "Strixhaven Skycoach",
    "Grapple with Death",
    "Pest Mascot",
    "Essenceknit Scholar",
    "Royal Treatment",
    "Last Gasp",
    "Teacher's Pest",
    "Mindful Biomancer",
    "Shared Roots",
    "Send in the Pest",
    "Pull from the Grave",
    "Follow the Lumarets",
    "Follow the Lumarets",
    "Slumbering Trudge",
    "Wander Off",
    "Stirring Hopesinger",
    "Bogwater Lumaret",
    "Forum Necroscribe",
    "Old-Growth Educator",
    "Forum of Amity",
    "Burrog Barrage",
    "Cost of Brilliance",
    "Lecturing Scornmage",
    "Chelonian Tackle",
    "Page, Loose Leaf",
    "Shopkeeper's Bane",
    "Last Gasp",
    "Strixhaven Skycoach",
    "Stirring Honormancer",
    "Choreographed Sparks",
    "Ral Zarek, Guest Lecturer",
    "Pest Mascot",
]

pick_options = choose_pick(
    pool_cards=pool,
    pack_cards=pack,
    pack_number=0,
    pick_number=2,
)
pick, score = pick_options[0]

print("Pick probabilities:")
for card, probability in pick_options:
    print(f"{card}: {probability:.3f}")

print(f"\nModel pick: {pick} ({score:.3f})")

if last_pick:
    if model_name != "draftscorer":
        print("This checkpoint does not support deckbuilding.")
    else:
        final_pool_counts = counts_from_names(pool)
        final_pool_counts[0, card_to_index[pick]] += 1

        target_size = max(FULL_DECK_SIZE, int(model_kwargs.get("target_deck_size", FULL_DECK_SIZE)))
        deck = build_deck(final_pool_counts, target_size=target_size)
        print_cards("Deck after draft", deck)
