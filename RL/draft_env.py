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

    
    def names_to_counts(self, card_list):
        counts = np.zeros(self.num_cards, dtype=np.float32)
        for card in card_list:
            counts[self.card_to_index[card]] += 1.0
        return counts

    def reset(self, seed=None, options=None):
        super().reset(seed=rand.seed)

      #  with open("/MTG_Draft_Agent/SL/data/raw/SOS_cards.json") as json_file:
            #json.dump(data, json_file, indent = 4)
        self.info = {}
        self.packs = []


        for item in ["1", "2", "3"]:
            pack = []

            pack.append(rand.choice(self.rare_mythic))
            pack.extend(rand.sample(self.uncommon, 3))    # sample = no duplicates
            pack.extend(rand.sample(self.common, 11))

            self.info[item] = pack
            self.packs.append(self.names_to_counts(pack))

        self.pool_counts = np.zeros(self.num_cards, dtype=np.float32)
        self.pack_number = 0
        self.pick_number = 0

        return self.get_observations(), {"action_mask": self.get_mask()}
    
    def get_observations(self):

        return {
    
            "pool_counts": self.pool_counts.copy(),
            "pack_counts": self.packs[self.pack_number].copy(),
            "pack_number": self.pack_number,
            "pick_number": self.pick_number
        }
    
    def get_mask(self):
        current_pack = self.packs[self.pack_number]
        return current_pack > 0

    def step(self, action):

        terminated = False
        current_pack = self.packs[self.pack_number]

        # Validate Action
        assert current_pack[action] > 0, "Invalid action: card not in current pack"

        # Mutate (remove card from card_indexes)
        current_pack[action] -= 1
        self.pool_counts[action] += 1

        # Advance 
        self.pick_number += 1

        if self.pick_number == 14:
            self.pick_number = 0
            self.pack_number += 1

        # Check termination
        if self.pack_number == 3:
            terminated = True

        # Reward function
        reward = 0

        # Return observation dict
        if terminated == True:
            observations = {
                "pool_counts": self.pool_counts.copy(),
                "pack_counts": np.zeros(self.num_cards, dtype=np.float32),
                "pack_number": 2,
                "pick_number": 14
            }

            info = {
                "action_mask": np.zeros(self.num_cards, dtype=bool),
                "final_pool": self.pool_counts.copy(),
               }
        else:
            observations = self.get_observations()
            info = {"action_mask": self.get_mask()}

        return observations, reward, terminated, False, info
        
        

    def render(self):

        pass

    def nameToIndex(self, card_name):
        return self.card_to_index.get(card_name, -1)
    
    def IndexToName(self, index):
        return self.index_to_card.get(index, "Card not found")
