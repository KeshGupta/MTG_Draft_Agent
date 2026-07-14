import pandas as pd
import json
import random as rand

names = []
common = []
uncommon = []
rare = []
mythic = []


df = pd.read_json("SOS_cards.json")

#print(df.columns)

names = df["name"].tolist()

intermediate = df.loc[(df["rarity"] == "common") & (df["mana_cost"] != "")]
common = intermediate["name"].tolist()

intermediate = df.loc[df["rarity"] == "uncommon"]
uncommon = intermediate["name"].tolist()

intermediate = df.loc[df["rarity"] == "rare"]
rare = intermediate["name"].tolist()

intermediate = df.loc[df["rarity"] == "mythic"]
mythic = intermediate["name"].tolist()

rare_mythic = rare + mythic
info = {}


for item in ["pack 1", "pack 2", "pack 3"]:

    pack = []

    rare_mythic_card = rand.choice(rare_mythic)
    pack.append(rare_mythic_card)

    for i in range(3):
        uncommon_card = rand.choice(uncommon)
        pack.append(uncommon_card)

    for i in range (11):
        common_card = rand.choice(common)
        pack.append(common_card)

    info[item] = pack

print(info)



