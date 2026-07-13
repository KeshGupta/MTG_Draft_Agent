import json


card_name = "Expressive Firedancer"
with open("C:\\Users\\samth\\OneDrive\\Desktop\\mtg\\MTG_Draft_Agent\\SL\\data\\cleaned\\metadata.json", encoding="utf-8") as f:
    meta = json.load(f)

card_names = meta["card_names"]
card_to_index = {name: i for i, name in enumerate(card_names)}

action_index = card_to_index.get(card_name, -1)

print(action_index)  # This will print the index of the card name or -1 if not found