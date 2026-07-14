import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
import json

class draft_env(gym.Env):
    def __init__(self, num_cards=346):
        super().__init__()

        self.action_space = spaces.Discrete(num_cards)  #num_cards cards to choose from

        C = num_cards
        observation_space = spaces.Dict({
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


    def reset(self):
      #  with open("/MTG_Draft_Agent/SL/data/raw/SOS_cards.json") as json_file:
            #json.dump(data, json_file, indent = 4)

        df = pd.read_json("/MTG_Draft_Agent/SL/data/raw/SOS_cards.json")
        print(df)

        return initial_state

    def step(self, action):

        return next_state, terminated, info
    
    def render(self):

        pass

    def nameToIndex(self, card_name):
        return self.card_to_index.get(card_name, -1)
    
    def IndexToName(self, index):
        return self.index_to_card.get(index, "Card not found")
