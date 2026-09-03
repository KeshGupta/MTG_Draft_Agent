import json

def create_initial_state(history):
    """
    Creates the initial state of the game.
    """
    initial_state = {
        "player 1": {
            "life": 20,
            "hand": [],
            "deck": [],
            "graveyard": [],
            "battlefield": [],
            "exile": [],
            "mana_pool": {
                "white": 0,
                "blue": 0,
                "black": 0,
                "red": 0,
                "green": 0,
                "colorless": 0
            }
        },
        "player 2": {
            "life": 20,
            "hand": [],
            "deck": [],
            "graveyard": [],
            "battlefield": [],
            "exile": [],
            "mana_pool": {
                "white": 0,
                "blue": 0,
                "black": 0,
                "red": 0,
                "green": 0,
                "colorless": 0
            }
        },
        "stack": [],
    }

    
    return initial_state



with open("MTG_Bot/pre_processing/data/raw_replays/5f2f3d857cae4325bcea3290f4d3964a/match_0_game_0/history.json", "r") as f:
    history = json.load(f)

initial_state = create_initial_state(history)