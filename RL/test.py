from pathlib import Path
from collections import Counter

import numpy as np

from SL.util.deckbuilding import BASIC_LANDS, COLORLESS_LAND, build_deck, load_ratings_table


ROOT = Path(__file__).resolve().parents[1]
RATINGS_CSV = ROOT / "card-ratings-2026-07-14 (1).csv"
CARDS_JSON = ROOT / "SOS_cards.json"


def make_sample_pool(table, index_to_card, cards_per_bucket=8):
    """Create a deterministic fake draft pool from rated cards."""
    pool_counts = np.zeros(len(index_to_card), dtype=np.float32)
    card_to_index = {name: index for index, name in index_to_card.items()}

    sample = (
        table.assign(bucket=table["cmc"].clip(upper=6))
        .sort_values(["bucket", "GIH WR"], ascending=[True, False])
        .groupby("bucket")
        .head(cards_per_bucket)
    )

    for card_name in sample["Name"]:
        if card_name in card_to_index:
            pool_counts[card_to_index[card_name]] += 1

    return pool_counts


def print_deck(deck, table):
    details = table.set_index("Name")
    land_names = set(BASIC_LANDS.values()) | {COLORLESS_LAND}
    spells = [card for card in deck if card not in land_names]
    lands = [card for card in deck if card in land_names]

    print(f"\nDeckbuilder chose {len(deck)} cards:\n")
    print("Spells:")

    for card_name in spells:
        row = details.loc[card_name]
        print(f"{int(row['cmc']):>2}  {row['GIH WR']:>5.1f}%  {card_name}")

    print("\nLands:")
    for land_name, count in sorted(Counter(lands).items()):
        print(f"{count:>2}  {land_name}")

    print("\nCurve:")
    for cmc, count in (
        details.loc[spells, "cmc"]
        .clip(upper=6)
        .astype(int)
        .value_counts()
        .sort_index()
        .items()
    ):
        label = "6+" if cmc == 6 else str(cmc)
        print(f"  {label}: {count}")


def main():
    table = load_ratings_table(RATINGS_CSV, CARDS_JSON)
    card_names = sorted(table["Name"].unique())
    index_to_card = dict(enumerate(card_names))

    pool_counts = make_sample_pool(table, index_to_card)
    deck = build_deck(pool_counts, index_to_card, table)

    print(f"Ratings rows loaded: {len(table)}")
    print(f"Sample pool size: {int(pool_counts.sum())}")
    print_deck(deck, table)

    land_names = set(BASIC_LANDS.values()) | {COLORLESS_LAND}
    spell_names = set(index_to_card.values()) - land_names
    spells = [card for card in deck if card in spell_names]
    lands = [card for card in deck if card in land_names]

    assert len(deck) == 40, f"Expected 40 cards, got {len(deck)}"
    assert len(spells) == 23, f"Expected 23 spells, got {len(spells)}"
    assert len(lands) == 17, f"Expected 17 lands, got {len(lands)}"
    assert all(card in spell_names or card in land_names for card in deck), "Deck contains unknown cards"


if __name__ == "__main__":
    main()
