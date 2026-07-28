import gymnasium as gym
import numpy as np
from gymnasium import spaces
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent           # RL/
ROOT = HERE.parent                                # project root
CARDS_JSON = ROOT / "SOS_cards.json"
EXTRA_CARDS_JSON = ROOT / "SL" / "data" / "raw" / "SOS_extra_cards.json"
METADATA_JSON = ROOT / "SL" / "data" / "cleaned_with_decks" / "metadata.json"
RATINGS_CSV = ROOT / "card-ratings-2026-07-14 (2).csv"
BASIC_LAND_NAMES = ("Plains", "Island", "Swamp", "Mountain", "Forest")
DEFAULT_TABLE_SIZE = 8
DEFAULT_TARGET_DECK_SIZE = 40
SPECIAL_GUEST_RATE = 1 / 55
RARE_SLOT_MYTHIC_RATE = 0.141 / (0.825 + 0.141)
COMMON_DUAL_LAND_RATE = 0.50
BOT_RANDOM_PICK_RATE = 0.08
BOT_RATING_TEMPERATURE = 0.025
UNRATED_CARD_SCORE = 0.45
WIN_RATE_RATING_COLUMNS = ("GIH WR", "GD WR", "OH WR", "GP WR", "GNS WR")
DRAFT_POSITION_COLUMNS = ("ATA", "ALSA")
DRAFT_POSITION_BEST_SCORE = 0.60
DRAFT_POSITION_WORST_SCORE = 0.40
DRAFT_POSITION_MAX = 14.0
WILDCARD_RARITY_WEIGHTS = {
    "common": 0.391,
    "uncommon": 0.391,
    "rare": 0.195,
    "mythic": 0.019,
}
MYSTICAL_ARCHIVE_RARITY_WEIGHTS = {
    "common": 0.125,
    "uncommon": 0.750,
    "rare": 0.096,
    "mythic": 0.029,
}
FOIL_RARITY_WEIGHTS = {
    "common": 0.544,
    "uncommon": 0.364,
    "rare": 0.072,
    "mythic": 0.015,
}

class draft_env(gym.Env):
    def __init__(
        self,
        num_cards=None,
        card_names=None,
        target_deck_size=DEFAULT_TARGET_DECK_SIZE,
        table_size=DEFAULT_TABLE_SIZE,
        ratings_path=RATINGS_CSV,
        bot_random_pick_rate=BOT_RANDOM_PICK_RATE,
        bot_rating_temperature=BOT_RATING_TEMPERATURE,
    ):
        super().__init__()

        # canonical card order: prefer the list passed in (from the checkpoint);
        # fall back to metadata.json only if none given.
        if card_names is None:
            with open(METADATA_JSON, encoding="utf-8") as f:
                card_names = json.load(f)["card_names"]

        self.card_to_index = {name: i for i, name in enumerate(card_names)}
        self.index_to_card = {i: name for i, name in enumerate(card_names)}

        self.num_cards = len(card_names) if num_cards is None else num_cards
        self.target_deck_size = int(target_deck_size)
        self.table_size = int(table_size)
        self.bot_random_pick_rate = float(bot_random_pick_rate)
        self.bot_rating_temperature = float(bot_rating_temperature)
        if self.target_deck_size <= 0:
            raise ValueError("target_deck_size must be positive.")
        if self.table_size < 2:
            raise ValueError("table_size must be at least 2.")
        if not 0.0 <= self.bot_random_pick_rate <= 1.0:
            raise ValueError("bot_random_pick_rate must be between 0 and 1.")
        if self.bot_rating_temperature <= 0:
            raise ValueError("bot_rating_temperature must be positive.")

        C = self.num_cards
        self.action_space = spaces.Discrete(C)
        self.observation_space = spaces.Dict({
            "pool_counts": spaces.Box(low=0, high=45, shape=(C,), dtype=np.float32),
            "pack_counts": spaces.Box(low=0, high=14, shape=(C,), dtype=np.float32),
            "deck_counts": spaces.Box(low=0, high=self.target_deck_size, shape=(C,), dtype=np.float32),
            "pack_number": spaces.Discrete(3),
            "pick_number": spaces.Discrete(14),
            "phase": spaces.Discrete(2),
            "build_step": spaces.Discrete(self.target_deck_size + 1),
        })

        self.pack_number = 0
        self.pick_number = 0
        self.phase = 0
        self.build_step = 0

        main_records = self._load_indexed_records(CARDS_JSON)
        extra_records = self._load_indexed_records(EXTRA_CARDS_JSON) if EXTRA_CARDS_JSON.exists() else []
        self.card_rating = self._load_card_ratings(Path(ratings_path))

        self.regular_commons = self._unique_names(
            card for card in main_records
            if card.get("rarity") == "common"
            and not self._is_land(card)
            and card.get("mana_cost")
        )
        self.uncommons = self._unique_names(card for card in main_records if card.get("rarity") == "uncommon")
        self.rares = self._unique_names(card for card in main_records if card.get("rarity") == "rare")
        self.mythics = self._unique_names(card for card in main_records if card.get("rarity") == "mythic")
        self.basic_lands = [name for name in BASIC_LAND_NAMES if name in self.card_to_index]
        self.common_lands = self._unique_names(
            card for card in main_records
            if card.get("rarity") == "common" and self._is_land(card) and not self._is_basic_land(card)
        )
        self.special_guests = self._unique_names(
            card for card in extra_records
            if not self._is_instant_or_sorcery(card)
        )
        self.wildcard_pools = {
            rarity: self._unique_names(
                card for card in main_records
                if card.get("rarity") == rarity and not self._is_basic_land(card)
            )
            for rarity in WILDCARD_RARITY_WEIGHTS
        }
        self.archive_pools = {
            rarity: self._unique_names(
                card for card in extra_records
                if card.get("rarity") == rarity and self._is_instant_or_sorcery(card)
            )
            for rarity in MYSTICAL_ARCHIVE_RARITY_WEIGHTS
        }
        if not any(self.archive_pools.values()):
            self.archive_pools = {
                rarity: self._unique_names(
                    card for card in main_records
                    if card.get("rarity") == rarity and self._is_instant_or_sorcery(card)
                )
                for rarity in MYSTICAL_ARCHIVE_RARITY_WEIGHTS
            }

        self._require_pool("regular commons", self.regular_commons)
        self._require_pool("uncommons", self.uncommons)
        self._require_pool("rares", self.rares)
        self._require_pool("mythics", self.mythics)
        self._require_pool("basic lands or common lands", self.basic_lands or self.common_lands)

    def _indexed_name(self, card_name):
        if card_name in self.card_to_index:
            return card_name
        for part in (card_name or "").split(" // "):
            if part in self.card_to_index:
                return part
        return None

    def _load_indexed_records(self, path):
        with open(path, encoding="utf-8") as file:
            raw_records = json.load(file)

        records = []
        for card in raw_records:
            indexed_name = self._indexed_name(card.get("name"))
            if indexed_name is None:
                continue
            record = dict(card)
            record["source_name"] = card.get("name")
            record["name"] = indexed_name
            records.append(record)
        return records

    @staticmethod
    def _parse_percent(value):
        text = str(value or "").strip().rstrip("%")
        if not text:
            return None
        try:
            return float(text) / 100.0
        except ValueError:
            return None

    @staticmethod
    def _parse_float(value):
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _draft_position_score(position):
        normalized = (position - 1.0) / (DRAFT_POSITION_MAX - 1.0)
        normalized = min(max(normalized, 0.0), 1.0)
        return DRAFT_POSITION_BEST_SCORE - normalized * (DRAFT_POSITION_BEST_SCORE - DRAFT_POSITION_WORST_SCORE)

    def _rating_score_from_row(self, row):
        for column in WIN_RATE_RATING_COLUMNS:
            score = self._parse_percent(row.get(column))
            if score is not None:
                return score

        position_scores = []
        for column in DRAFT_POSITION_COLUMNS:
            position = self._parse_float(row.get(column))
            if position is not None:
                position_scores.append(self._draft_position_score(position))
        if position_scores:
            return float(sum(position_scores) / len(position_scores))
        return None

    def _load_card_ratings(self, path):
        ratings = {}
        if not path.exists():
            return ratings

        with open(path, newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                card_name = self._indexed_name(row.get("Name"))
                if card_name is None:
                    continue
                score = self._rating_score_from_row(row)
                if score is not None:
                    ratings[card_name] = score
        return ratings

    @staticmethod
    def _is_land(card):
        return "Land" in (card.get("type_line") or "")

    @staticmethod
    def _is_basic_land(card):
        return "Basic Land" in (card.get("type_line") or "")

    @staticmethod
    def _is_instant_or_sorcery(card):
        type_line = card.get("type_line") or ""
        return "Instant" in type_line or "Sorcery" in type_line

    @staticmethod
    def _unique_names(cards):
        return sorted({card["name"] for card in cards if card.get("name")})

    @staticmethod
    def _require_pool(label, pool):
        if not pool:
            raise ValueError(f"No indexed cards available for booster slot: {label}.")

    def _choice(self, pool):
        self._require_pool("choice", pool)
        return str(self.np_random.choice(np.asarray(pool, dtype=object)))

    def _sample(self, pool, count):
        self._require_pool("sample", pool)
        replace = len(pool) < count
        return self.np_random.choice(np.asarray(pool, dtype=object), size=count, replace=replace).astype(str).tolist()

    def _weighted_choice(self, pools, weights):
        available = [(rarity, weight) for rarity, weight in weights.items() if pools.get(rarity)]
        if not available:
            fallback = [card for pool in pools.values() for card in pool]
            return self._choice(fallback)

        rarities, raw_weights = zip(*available)
        probabilities = np.asarray(raw_weights, dtype=np.float64)
        probabilities /= probabilities.sum()
        rarity = str(self.np_random.choice(np.asarray(rarities, dtype=object), p=probabilities))
        return self._choice(pools[rarity])

    def _rare_or_mythic(self):
        if self.mythics and (not self.rares or self.np_random.random() < RARE_SLOT_MYTHIC_RATE):
            return self._choice(self.mythics)
        return self._choice(self.rares)

    def _land_slot(self):
        if self.common_lands and (not self.basic_lands or self.np_random.random() < COMMON_DUAL_LAND_RATE):
            return self._choice(self.common_lands)
        return self._choice(self.basic_lands or self.common_lands)

    def _make_booster(self):
        pack = self._sample(self.regular_commons, 6)
        if self.special_guests and self.np_random.random() < SPECIAL_GUEST_RATE:
            pack[-1] = self._choice(self.special_guests)

        pack.extend(self._sample(self.uncommons, 3))
        pack.append(self._weighted_choice(self.wildcard_pools, WILDCARD_RARITY_WEIGHTS))
        pack.append(self._rare_or_mythic())
        pack.append(self._weighted_choice(self.archive_pools, MYSTICAL_ARCHIVE_RARITY_WEIGHTS))
        pack.append(self._weighted_choice(self.wildcard_pools, FOIL_RARITY_WEIGHTS))
        pack.append(self._land_slot())
        return pack

    def _start_draft_round(self):
        self.current_packs = [self._make_booster() for _ in range(self.table_size)]
        self.info[f"round_{self.pack_number + 1}_opened_pack"] = self.current_packs[0].copy()

    def _start_build_phase(self):
        self.phase = 1
        self.pack_number = 2
        self.pick_number = 13
        self.build_step = 0
        self.final_pool_counts = self.pool_counts.copy()
        self.deck_counts = np.zeros(self.num_cards, dtype=np.float32)
        self.current_packs = [[] for _ in range(self.table_size)]

    def _pass_left(self):
        return self.pack_number != 1

    def _rotate_packs(self):
        if self._pass_left():
            self.current_packs = [self.current_packs[-1]] + self.current_packs[:-1]
        else:
            self.current_packs = self.current_packs[1:] + [self.current_packs[0]]

    def _bot_pick_index(self, pack):
        if self.np_random.random() < self.bot_random_pick_rate:
            return int(self.np_random.integers(len(pack)))

        scores = np.asarray(
            [self.card_rating.get(card, UNRATED_CARD_SCORE) for card in pack],
            dtype=np.float64,
        )
        logits = (scores - scores.max()) / self.bot_rating_temperature
        weights = np.exp(logits)
        weights /= weights.sum()
        return int(self.np_random.choice(np.arange(len(pack)), p=weights))

    def _deck_nametext(self):
        lines = []
        for index, count in enumerate(self.deck_counts.astype(int)):
            if count > 0:
                lines.append(f"{count} {self.index_to_card[index]}")
        return "\n".join(lines)

    
    def names_to_counts(self, card_list):
        counts = np.zeros(self.num_cards, dtype=np.float32)
        for card in card_list:
            if card in self.card_to_index:
                counts[self.card_to_index[card]] += 1.0
        return counts

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.info = {}
        self.pool_counts = np.zeros(self.num_cards, dtype=np.float32)
        self.deck_counts = np.zeros(self.num_cards, dtype=np.float32)
        self.final_pool_counts = None
        self.pack_number = 0
        self.pick_number = 0
        self.phase = 0
        self.build_step = 0
        self._start_draft_round()

        return self.get_observations(), self._info()
    
    def get_observations(self):
        pack_counts = (
            self.names_to_counts(self.current_packs[0])
            if self.phase == 0
            else np.zeros(self.num_cards, dtype=np.float32)
        )
        return {
            "pool_counts": self.pool_counts.copy(),
            "pack_counts": pack_counts,
            "deck_counts": self.deck_counts.copy(),
            "pack_number": self.pack_number,
            "pick_number": self.pick_number,
            "phase": self.phase,
            "build_step": self.build_step,
        }
    
    def get_mask(self):
        if self.phase == 0:
            return self.names_to_counts(self.current_packs[0]) > 0

        basic_mask = np.zeros(self.num_cards, dtype=bool)
        for land_name in self.basic_lands:
            basic_mask[self.card_to_index[land_name]] = True
        return (self.pool_counts > self.deck_counts) | basic_mask

    def step(self, action):
        action = int(action)
        if action < 0 or action >= self.num_cards:
            raise ValueError(f"Invalid action index {action}.")
        if self.phase == 0:
            return self._step_draft(action)
        return self._step_build(action)

    def _step_draft(self, action):
        mask = self.get_mask()
        if not bool(mask[action]):
            raise ValueError(f"Invalid draft action: {self.index_to_card[action]} is not in the current pack.")

        picked_name = self.index_to_card[action]
        self.current_packs[0].remove(picked_name)
        self.pool_counts[action] += 1

        for seat in range(1, self.table_size):
            if self.current_packs[seat]:
                bot_pick = self._bot_pick_index(self.current_packs[seat])
                self.current_packs[seat].pop(bot_pick)

        self.pick_number += 1

        if self.pick_number == 14:
            self.pick_number = 0
            self.pack_number += 1
            if self.pack_number == 3:
                self._start_build_phase()
            else:
                self._start_draft_round()
        else:
            self._rotate_packs()

        return self.get_observations(), 0.0, False, False, self._info()

    def _step_build(self, action):
        mask = self.get_mask()
        if not bool(mask[action]):
            raise ValueError(f"Invalid build action: cannot add {self.index_to_card[action]}.")

        self.deck_counts[action] += 1
        self.build_step += 1
        terminated = int(self.deck_counts.sum()) >= self.target_deck_size
        return self.get_observations(), 0.0, terminated, False, self._info(terminated=terminated)

    def _info(self, terminated=False):
        info = {
            "action_mask": np.zeros(self.num_cards, dtype=bool) if terminated else self.get_mask(),
            "phase": "build" if self.phase == 1 else "draft",
            "final_pool": self.final_pool_counts.copy() if self.final_pool_counts is not None else None,
        }
        if terminated:
            deck_counts = self.deck_counts.copy()
            info.update({
                "completed_deck_counts": deck_counts,
                "completed_deck_names": [
                    self.index_to_card[index]
                    for index, count in enumerate(deck_counts.astype(int))
                    for _ in range(count)
                ],
                "completed_deck_nametext": self._deck_nametext(),
                "deck_nametext": self._deck_nametext(),
            })
        return info
        
        

    def render(self):

        pass

    def nameToIndex(self, card_name):
        return self.card_to_index.get(card_name, -1)
    
    def IndexToName(self, index):
        return self.index_to_card.get(index, "Card not found")
