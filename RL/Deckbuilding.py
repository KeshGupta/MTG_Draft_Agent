import pandas as pd
import numpy as np
import json

# --- model index -> card name (from your env's metadata) ---
with open("/Users/keshgupta/Documents/MTG_Draft_Agent/metadata.json", encoding="utf-8") as f:
    meta = json.load(f)
index_to_card = {i: name for i, name in enumerate(meta["card_names"])}

info = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70])

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
table = table.sort_values("games", ascending=False).drop_duplicates(subset="Name", keep="first")

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
