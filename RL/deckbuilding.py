import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
import json
import random as rand
from pathlib import Path

HERE = Path(__file__).resolve().parent           # RL/
ROOT = HERE.parent                                # project root
CARDS_JSON = ROOT / "data" / "SOS_cards.json"

class draft_env(gym.Env):
    def __init__(self, num_cards=None, card_names=None):
        super().__init__()

        # canonical card order: prefer the list passed in (from the checkpoint);
        # fall back to metadata.json only if none given.
        if card_names is None:
            with open(ROOT / "SL" / "data" / "cleaned" / "metadata.json", encoding="utf-8") as f:
                card_names = json.load(f)["card_names"]

        self.card_to_index = {name: i for i, name in enumerate(card_names)}
        self.index_to_card = {i: name for i, name in enumerate(card_names)}

        self.num_cards = len(card_names) if num_cards is None else num_cards
        C = self.num_cards
        self.action_space = spaces.Discrete(C)
        self.observation_space = spaces.Dict({
            "pool_counts": spaces.Box(low=0, high=4, shape=(C,), dtype=np.float32),
            "pack_counts": spaces.Box(low=0, high=4, shape=(C,), dtype=np.float32),
            "pack_number": spaces.Discrete(3),
            "pick_number": spaces.Discrete(14),
        })

        self.pack_number = 0
        self.pick_number = 0

        df = pd.read_json(CARDS_JSON)
        common   = df.loc[(df["rarity"] == "common") & (df["mana_cost"] != ""), "name"].tolist()
        uncommon = df.loc[df["rarity"] == "uncommon", "name"].tolist()
        rare     = df.loc[df["rarity"] == "rare", "name"].tolist()
        mythic   = df.loc[df["rarity"] == "mythic", "name"].tolist()

        self.common = common
        self.uncommon = uncommon
        self.rare_mythic = rare + mythic