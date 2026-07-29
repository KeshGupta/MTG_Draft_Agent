# deckbuilding.py - now a function, conversion at the top
import pandas as pd
import numpy as np
import re


BASIC_LANDS = {
    "W": "Plains",
    "U": "Island",
    "B": "Swamp",
    "R": "Mountain",
    "G": "Forest",
}
COLOR_ORDER = ["W", "U", "B", "R", "G"]
COLORLESS_LAND = "Wastes"

def load_ratings_table(ratings_path, cards_path, min_games=200):
    """Build the name-to-(GIH WR, cmc) table once, at startup."""
    ratings = pd.read_csv(ratings_path)
    cards   = pd.read_json(cards_path)
    ratings = ratings[["Name", "Color", "GIH WR", "# GIH"]].copy()
    ratings["GIH WR"] = ratings["GIH WR"].str.rstrip("%").astype(float)
    ratings = ratings.rename(columns={"# GIH": "games"})
    ratings = ratings.sort_values("games", ascending=False).drop_duplicates(subset="Name", keep="first")
    cards = cards[["name", "cmc", "mana_cost", "type_line"]].copy()
    table = cards.merge(ratings, left_on="name", right_on="Name", how="outer")
    table["Name"] = table["Name"].fillna(table["name"])
    table["Color"] = table["Color"].fillna("")
    table["games"] = table["games"].fillna(0).astype(int)
    table["GIH WR"] = table["GIH WR"].where(table["games"] >= min_games, 0.0).fillna(0.0)
    table["cmc"] = table["cmc"].fillna(3).astype(int)
    table["mana_cost"] = table["mana_cost"].fillna("")
    table["type_line"] = table["type_line"].fillna("")
    table["is_land"] = table["type_line"].str.contains("Land", na=False)
    table.loc[table["Name"].isin(BASIC_LANDS.values()), "is_land"] = True
    table = table.drop(columns=["name"])
    table = table.sort_values("games", ascending=False).drop_duplicates(subset="Name", keep="first")
    return table


def count_color_pips(spells, table):
    pips = {color: 0 for color in COLOR_ORDER}
    details = table.set_index("Name")

    for card_name in spells:
        if card_name not in details.index:
            continue
        mana_cost = str(details.loc[card_name, "mana_cost"])
        for symbol in re.findall(r"\{([^}]+)\}", mana_cost):
            for color in COLOR_ORDER:
                if color in symbol:
                    pips[color] += 1

    return pips


def add_basic_lands(spells, table, total_lands=17):
    pips = count_color_pips(spells, table)
    active_pips = {color: count for color, count in pips.items() if count > 0}

    if not active_pips:
        return spells + [COLORLESS_LAND] * total_lands

    total_pips = sum(active_pips.values())
    land_counts = {
        color: max(1, int(total_lands * count / total_pips))
        for color, count in active_pips.items()
    }

    while sum(land_counts.values()) < total_lands:
        color = max(active_pips, key=lambda c: active_pips[c] / (land_counts[c] + 1))
        land_counts[color] += 1

    while sum(land_counts.values()) > total_lands:
        color = max(
            (c for c in land_counts if land_counts[c] > 1),
            key=lambda c: land_counts[c] / active_pips[c],
        )
        land_counts[color] -= 1

    lands = []
    for color in COLOR_ORDER:
        lands.extend([BASIC_LANDS[color]] * land_counts.get(color, 0))
    return spells + lands


def build_deck(final_pool, index_to_card, table, add_lands=True, total_lands=17):
    # ---- Seam B: vector to names (the ONLY conversion in here) ----
    details = table.set_index("Name", drop=False)
    pool_rows = []
    for card_index, count in enumerate(final_pool):
        card_name = index_to_card[card_index]
        if count <= 0 or card_name not in details.index:
            continue
        row = details.loc[card_name]
        if bool(row["is_land"]):
            continue
        for _ in range(int(count)):
            pool_rows.append(row.to_dict())

    pool = pd.DataFrame(pool_rows)
    # -------------------------------------------------------------

    if pool.empty:
        return add_basic_lands([], table, total_lands) if add_lands else []

    pool["bucket"] = pool["cmc"].clip(upper=6)
    CARD_DIST = {1: 2, 2: 7, 3: 6, 4: 4, 5: 3, 6: 1}
    pool = pool.sort_values(["GIH WR", "games"], ascending=False).reset_index(drop=True)

    deck_indices = []
    for cmc, need in CARD_DIST.items():
        cands = pool[(pool["bucket"] == cmc) & (~pool.index.isin(deck_indices))]
        deck_indices.extend(cands.head(need).index.tolist())
    if len(deck_indices) < 23:
        remaining = pool[~pool.index.isin(deck_indices)].sort_values(["GIH WR", "games"], ascending=False)
        deck_indices.extend(remaining.head(23 - len(deck_indices)).index.tolist())

    deck = pool.loc[deck_indices, "Name"].tolist()

    if add_lands:
        deck = add_basic_lands(deck, table, total_lands)
    return deck
