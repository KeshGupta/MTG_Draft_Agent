import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
import json
import random as rand

class draft_env(gym.Env):
    def __init__(self, num_cards=346):
        super().__init__()

        self.action_space = spaces.Discrete(num_cards)  #num_cards cards to choose from
        self.num_cards = num_cards

        C = num_cards
        self.observation_space = spaces.Dict({
        "pool_counts": spaces.Box(low=0, high=4, shape=(C,), dtype=np.float32),
        "pack_counts": spaces.Box(low=0, high=4, shape=(C,), dtype=np.float32),
        "pack_number": spaces.Discrete(3),  
        "pick_number": spaces.Discrete(14),
        })

        self.pool = []
        self.pack = []

        self.pack_number = 0
        self.pick_number = 0

        #load card names in the order that they are in the nueral network output layer
        with open("C:\\Users\\samth\\OneDrive\\Desktop\\mtg\\MTG_Draft_Agent\\SL\\data\\cleaned\\metadata.json", encoding="utf-8") as f:
            meta = json.load(f)

        card_names = meta["card_names"]
        self.card_to_index = {name: i for i, name in enumerate(card_names)}
        self.index_to_card = {index: name for index, name in enumerate(card_names)}

        df = pd.read_json("SOS_cards.json")

        #names = df["name"].tolist()
        intermediate = df.loc[(df["rarity"] == "common") & (df["mana_cost"] != "")]
        self.common = intermediate["name"].tolist()

        intermediate = df.loc[df["rarity"] == "uncommon"]
        self.uncommon = intermediate["name"].tolist()

        intermediate = df.loc[df["rarity"] == "rare"]
        rare = intermediate["name"].tolist()

        intermediate = df.loc[df["rarity"] == "mythic"]
        mythic = intermediate["name"].tolist()

        self.rare_mythic = rare + mythic

        return 
    
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

            info = {}
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
