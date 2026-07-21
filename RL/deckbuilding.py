# deckbuilding.py  — now a function, conversion at the top
import pandas as pd
import numpy as np

def load_ratings_table(ratings_path, cards_path, min_games=200):
    """Build the name→(GIH WR, cmc) table once, at startup."""
    ratings = pd.read_csv(ratings_path)
    cards   = pd.read_json(cards_path)
    ratings = ratings[["Name", "Color", "GIH WR", "# GIH"]].copy()
    ratings["GIH WR"] = ratings["GIH WR"].str.rstrip("%").astype(float)
    ratings = ratings.rename(columns={"# GIH": "games"})
    cards = cards[["name", "cmc"]].copy()
    table = ratings.merge(cards, left_on="Name", right_on="name", how="inner").drop(columns=["name"])
    table["cmc"] = table["cmc"].astype(int)
    table = table[table["games"] >= min_games]
    table = table.sort_values("games", ascending=False).drop_duplicates(subset="Name", keep="first")
    return table


def build_deck(final_pool, index_to_card, table):
    # ---- Seam B: vector → names (the ONLY conversion in here) ----
    pool_idx = np.nonzero(final_pool)[0]
    pool_names = [index_to_card[i] for i in pool_idx]
    pool = table[table["Name"].isin(pool_names)].copy()
    # -------------------------------------------------------------

    pool["bucket"] = pool["cmc"].clip(upper=6)
    CARD_DIST = {1: 2, 2: 7, 3: 6, 4: 4, 5: 3, 6: 1}
    pool = pool.sort_values("GIH WR", ascending=False)

    deck = []
    for cmc, need in CARD_DIST.items():
        cands = pool[(pool["bucket"] == cmc) & (~pool["Name"].isin(deck))]
        deck.extend(cands["Name"].head(need).tolist())
        
    if len(deck) < 23:
        remaining = pool[~pool["Name"].isin(deck)].sort_values("GIH WR", ascending=False)
        deck.extend(remaining["Name"].head(23 - len(deck)).tolist())
    return deck