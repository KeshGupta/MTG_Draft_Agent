import pandas as pd
import numpy as np
from collections import Counter

BASICS = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}


def load_ratings_table(ratings_path, cards_path, min_games=200):
    """Build the name→(GIH WR, cmc, colors) table once, at startup."""
    ratings = pd.read_csv(ratings_path)
    cards   = pd.read_json(cards_path)
    ratings = ratings[["Name", "Color", "GIH WR", "# GIH"]].copy()
    ratings["GIH WR"] = ratings["GIH WR"].str.rstrip("%").astype(float)
    ratings = ratings.rename(columns={"# GIH": "games", "Color": "colors"})
    ratings["colors"] = ratings["colors"].fillna("")
    cards = cards[["name", "cmc"]].copy()
    table = ratings.merge(cards, left_on="Name", right_on="name", how="inner").drop(columns=["name"])
    table["cmc"] = table["cmc"].astype(int)
    table = table[table["games"] >= min_games]
    table = table.sort_values("games", ascending=False).drop_duplicates(subset="Name", keep="first")
    return table


def pick_deck_colors(pool):
    """Two dominant colors, weighted by GIH WR so the best cards decide."""
    weight = Counter()
    for _, row in pool.iterrows():
        for c in row["colors"]:
            weight[c] += row["GIH WR"]
    return set(c for c, _ in weight.most_common(2))


def in_colors(card_colors, deck_colors):
    return set(card_colors).issubset(deck_colors)


def land_split(chosen_rows, deck_colors, total_lands=17):
    """Split total_lands across deck_colors, weighted by colored pips in the chosen spells."""
    pips = Counter()
    for colors in chosen_rows:
        for c in colors:
            if c in deck_colors:
                pips[c] += 1

    # no colored pips (all colorless picks) → split evenly
    if not pips:
        colors = sorted(deck_colors) or ["W"]
        base = total_lands // len(colors)
        counts = {c: base for c in colors}
        for i in range(total_lands - base * len(colors)):
            counts[colors[i]] += 1
        return counts

    total_pips = sum(pips.values())
    raw = {c: total_lands * n / total_pips for c, n in pips.items()}
    floored = {c: int(raw[c]) for c in raw}

    # hand out leftover lands by largest fractional remainder
    remaining = total_lands - sum(floored.values())
    frac_order = sorted(raw, key=lambda c: raw[c] - floored[c], reverse=True)
    for i in range(remaining):
        floored[frac_order[i % len(frac_order)]] += 1

    # floor of 2 for any color actually used, so a splash isn't uncastable
    for c in list(floored):
        if pips[c] > 0 and floored[c] < 2:
            floored[c] = 2
    over = sum(floored.values()) - total_lands
    while over > 0:
        biggest = max(floored, key=lambda c: floored[c])
        floored[biggest] -= 1
        over -= 1

    return floored


def build_deck(final_pool, index_to_card, table, total_lands=17):
    # ---- Seam B: vector → names ----
    pool_idx = np.nonzero(final_pool)[0]
    pool_names = [index_to_card[i] for i in pool_idx]
    pool = table[table["Name"].isin(pool_names)].copy()

    # ---- color detection + filter ----
    deck_colors = pick_deck_colors(pool)
    pool = pool[pool["colors"].apply(lambda c: in_colors(c, deck_colors))].copy()

    # ---- curve-quota spell selection ----
    pool["bucket"] = pool["cmc"].clip(upper=6)
    CARD_DIST = {1: 2, 2: 7, 3: 6, 4: 4, 5: 3, 6: 1}   # 23 spells
    pool = pool.sort_values("GIH WR", ascending=False)

    spells = []
    for cmc, need in CARD_DIST.items():
        cands = pool[(pool["bucket"] == cmc) & (~pool["Name"].isin(spells))]
        spells.extend(cands["Name"].head(need).tolist())
    if len(spells) < 23:
        remaining = pool[~pool["Name"].isin(spells)].sort_values("GIH WR", ascending=False)
        spells.extend(remaining["Name"].head(23 - len(spells)).tolist())

    # ---- lands weighted by pips in the chosen spells ----
    chosen_colors = pool[pool["Name"].isin(spells)]["colors"].tolist()
    lands = land_split(chosen_colors, deck_colors, total_lands)

    # ---- emit full 40-card deck in "{count} {name}" .dck format ----
    spell_counts = Counter(spells)
    deck_lines = [f"{n} {name}" for name, n in spell_counts.items()]
    deck_lines += [f"{count} {BASICS[color]}" for color, count in sorted(lands.items())]
    return deck_lines