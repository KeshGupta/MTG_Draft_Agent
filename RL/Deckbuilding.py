import pandas as pd
import numpy as np
import json

# --- model index -> card name (from your env's metadata) ---
with open("/Users/keshgupta/Documents/MTG_Draft_Agent/metadata.json", encoding="utf-8") as f:
    meta = json.load(f)
index_to_card = {i: name for i, name in enumerate(meta["card_names"])}

info = np.array([0, 44, 88, 145, 13, 2, 17, 24, 40, 80, 300, 11, 12, 13,
                 45, 222, 55, 34, 67, 89, 90, 91, 92, 93, 94, 95, 96, 97,
                 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108])

ratings = pd.read_csv("/Users/keshgupta/Documents/MTG_Draft_Agent/card-ratings-2026-07-14 (1).csv")
cards   = pd.read_json("/Users/keshgupta/Documents/MTG_Draft_Agent/SOS_cards.json")

# clean GIH WR to float, keep games-seen for a reliability filter
ratings = ratings[["Name", "Color", "GIH WR", "# GIH"]].copy()
ratings["GIH WR"] = ratings["GIH WR"].str.rstrip("%").astype(float)
ratings = ratings.rename(columns={"# GIH": "games"})

cards = cards[["name", "cmc"]].copy()

# one clean merge instead of nested loops
table = ratings.merge(cards, left_on="Name", right_on="name", how="inner")
table = table.drop(columns=["name"])
table["cmc"] = table["cmc"].astype(int)

# min-games reliability filter (tune threshold)
MIN_GAMES = 200
table = table[table["games"] >= MIN_GAMES]

# --- restrict to the drafted pool, via model index -> name ---
pool_names = [index_to_card[idx] for idx in info]
pool = table[table["Name"].isin(pool_names)].copy()

# cmc 6+ collapses into one bucket
pool["bucket"] = pool["cmc"].clip(upper=6)

CARD_DIST = {1: 2, 2: 7, 3: 6, 4: 4, 5: 3, 6: 1}   # sums to 23

pool = pool.sort_values(["GIH WR"], ascending=False)
deck = []

# primary pass: fill each bucket from its own CMC, best GIH WR first
for cmc, need in CARD_DIST.items():
    cands = pool[(pool["bucket"] == cmc) & (~pool["Name"].isin(deck))]
    deck.extend(cands["Name"].head(need).tolist())

# backfill: if short of 23, take best remaining cards regardless of bucket
if len(deck) < 23:
    remaining = pool[~pool["Name"].isin(deck)].sort_values("GIH WR", ascending=False)
    deck.extend(remaining["Name"].head(23 - len(deck)).tolist())

print(len(deck), deck)
