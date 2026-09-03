#!/usr/bin/env python3
"""Turn a raw MTG Arena game log (history.json + its cards.json) into flat,
per-decision (state, legal_mask, action) training examples - the richer
sibling of cleanReplayData.py, built from the actual client/server protocol
stream instead of 17Lands' aggregated CSV.

Input is one game "bundle": history.json (the GRE diff stream), cards.json
(this bundle's own grpId -> name/types/abilities table - self-sufficient,
no dependency on SOS_cards.json), and optionally history_info.json (deck
context, kept for provenance only). manifest.json's `files` mapping is the
usual way these are named together in one folder.

Output uses the SAME phase ids, action space (num_cards + 1, last slot is
STOP), and self_*/opp_* perspective convention as cleanReplayData.py's
output, specifically so the two are compatible/mergeable for training:
  PHASE_LAND=2, PHASE_SPELL=3, PHASE_ATTACK=4, PHASE_BLOCK=5

What's different, and why, is explained in the design discussion this was
built from. Two properties worth knowing before reading the code:

1. legal_mask is always exactly what the game engine itself offered as
   candidates (the `actions` field) - never hand-computed. That means
   summoning sickness, mana availability for casting, vigilance, etc. are
   all correctly reflected in legal_mask for free, by construction, without
   this script having to get MTG's legality rules right. The extra state
   features this script *does* compute by hand (tapped/sick splits,
   available mana) are best-effort context for the model to learn from,
   not correctness-critical - if they're slightly off, legality isn't
   compromised, only how much the model can pick up about *why* a line
   is legal.
2. There's no explicit "chosen action" event in this log - only which
   candidates were offered, and (for a handful of target/order choices)
   ClientMessageType_SelectNResp answers. Land/spell/attack/block choices
   are inferred by watching a decision's candidate set shrink from one
   state to the next: the candidate that disappears is what was picked.
   Every step this produces is corroborated by a second, independent
   signal (a zone or tapped-state change matching that decision type) and
   the corroboration rate is written to metadata.json - that number is
   the honest confidence check on this whole approach, not a claim of
   correctness from me.

Deliberately deferred (see the design discussion): targeting and trigger/
effect ordering (ClientMessageType_SelectNResp) are detected and counted
but not yet turned into training rows - there's no "request" event paired
with them in this log, so attributing a response to the spell/ability it
answers needs more games to cross-validate before committing to an
encoding. Mulligan decisions are out of scope for the same reason: a
different shape of decision, not part of "what to do this turn."
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = ROOT / "SL" / "data" / "cleaned_with_decks" / "metadata.json"
DEFAULT_OUTPUT_DIR = ROOT / "SL" / "data" / "cleaned_history"

STARTING_LIFE = 20.0

# Same phase ids as cleanReplayData.py so the two outputs are compatible.
PHASE_LAND, PHASE_SPELL, PHASE_ATTACK, PHASE_BLOCK = 2, 3, 4, 5
PHASE_NAMES = {PHASE_LAND: "land", PHASE_SPELL: "spell", PHASE_ATTACK: "attack", PHASE_BLOCK: "block"}

MANA_SYMBOLS = ["W", "U", "B", "R", "G"]  # + a trailing "generic" slot appended in code
MANA_SYMBOL_RE = re.compile(r"\{o([WUBRG])\}")

BATTLEFIELD_LIKE = {"ZoneType_Battlefield"}
HAND_LIKE = {"ZoneType_Hand"}
STACK_LIKE = {"ZoneType_Stack"}
GRAVEYARD_LIKE = {"ZoneType_Graveyard"}
EXILE_LIKE = {"ZoneType_Exile"}


# --------------------------------------------------------------------------
# Card resolution: this bundle's own cards.json, not SOS_cards.json
# --------------------------------------------------------------------------

def load_cards_bundle(cards_json_path: Path) -> dict[int, dict[str, Any]]:
    """grpId -> {name, types, abilities_text, keyword_text} for this bundle.
    Keyword names show up inconsistently as bare values in either
    "abilities" or "hidden_abilities" (e.g. Flying/Haste sometimes sit
    directly in "abilities" alongside full rules text, sometimes only in
    "hidden_abilities") - keyword_text unions both so a simple substring
    check covers either case."""
    raw = json.loads(cards_json_path.read_text(encoding="utf-8"))
    by_id: dict[int, dict[str, Any]] = {}
    for card in raw.get("cards", []):
        abilities = card.get("abilities") or {}
        hidden = card.get("hidden_abilities") or {}
        by_id[int(card["id"])] = {
            "name": card.get("name") or "Unknown",
            "types": card.get("types") or [],
            "abilities_text": " ".join(abilities.values()),
            "keyword_text": " ".join(list(abilities.values()) + list(hidden.values())),
        }
    return by_id


def has_keyword(grp_id: int, keyword: str, cards_by_id: dict[int, dict[str, Any]]) -> bool:
    text = cards_by_id.get(grp_id, {}).get("keyword_text", "")
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def load_card_index(metadata_path: Path) -> tuple[list[str], dict[str, int]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    card_names = metadata["card_names"]
    return card_names, {name: index for index, name in enumerate(card_names)}


def build_grp_id_index(cards_by_id: dict[int, dict[str, Any]], card_to_index: dict[str, int]) -> tuple[dict[int, int], int, int]:
    """grpId -> this project's card index, via name matching against the
    same metadata.json used everywhere else."""
    grp_to_index: dict[int, int] = {}
    seen = resolved = 0
    for grp_id, card in cards_by_id.items():
        name = card["name"]
        if name in ("Unknown", ""):
            continue
        seen += 1
        index = card_to_index.get(name)
        if index is None:
            for part in name.split(" // "):
                index = card_to_index.get(part)
                if index is not None:
                    break
        if index is not None:
            grp_to_index[grp_id] = index
            resolved += 1
    return grp_to_index, seen, resolved


def mana_colors_for(grp_id: int, cards_by_id: dict[int, dict[str, Any]]) -> set[str]:
    """Colors a permanent's own abilities can add, parsed from its ability
    text (e.g. "{oT}: Add {oR}." -> {"R"}) rather than decoding the
    ColorProduction annotation's internal enum, which isn't documented
    anywhere I could verify against. A land that can add more than one
    color (e.g. a dual) contributes to every color it lists - this is a
    "could contribute" superset, not "will produce N mana simultaneously"."""
    text = cards_by_id.get(grp_id, {}).get("abilities_text", "")
    return set(MANA_SYMBOL_RE.findall(text))


# --------------------------------------------------------------------------
# State reconstruction: apply the GRE diff stream cumulatively
# --------------------------------------------------------------------------

class GameStateTracker:
    """Replays GameStateMessage diffs into a live view of objects, zones,
    players and turn info. Call .apply(gameStateMessage) once per event, in
    order."""

    def __init__(self) -> None:
        self.objects: dict[int, dict[str, Any]] = {}
        self.zones: dict[int, dict[str, Any]] = {}
        self.players: dict[int, dict[str, Any]] = {}
        self.turn_info: dict[str, Any] = {}
        self.game_over: bool = False
        self.entered_bf_turn: dict[int, int] = {}
        self._id_remap: dict[int, int] = {}  # historical id -> current id

    def resolve_id(self, instance_id: int) -> int:
        """Follow ObjectIdChanged remaps to the current id for a permanent
        whose identity may have changed zones/ids since it entered play."""
        seen = set()
        while instance_id in self._id_remap and instance_id not in seen:
            seen.add(instance_id)
            instance_id = self._id_remap[instance_id]
        return instance_id

    def zone_type(self, zone_id: int) -> str | None:
        zone = self.zones.get(zone_id)
        return zone.get("type") if zone else None

    def hand_zone_id(self, seat_id: int) -> int | None:
        for zone_id, zone in self.zones.items():
            if zone.get("type") in HAND_LIKE and zone.get("ownerSeatId") == seat_id:
                return zone_id
        return None

    def apply(self, gsm: dict[str, Any]) -> None:
        if gsm.get("gameInfo", {}).get("stage") == "GameStage_GameOver":
            # Seen on an event whose turnInfo is absent, so phase/step stay
            # stale at "still in the last real phase" and `actions` going
            # empty right after would otherwise look exactly like a real
            # elimination (everything still offered simultaneously vanishing)
            # instead of the game just ending. This flag is the actual
            # closing signal - checked by the caller before touching actions.
            self.game_over = True

        for zone in gsm.get("zones", []) or []:
            self.zones.setdefault(zone["zoneId"], {}).update(zone)

        turn = gsm.get("turnInfo")
        if turn:
            self.turn_info.update(turn)

        for player in gsm.get("players", []) or []:
            seat = player.get("systemSeatNumber")
            if seat is not None:
                self.players.setdefault(seat, {}).update(player)

        turn_number = self.turn_info.get("turnNumber")
        for obj in gsm.get("gameObjects", []) or []:
            instance_id = obj["instanceId"]
            previous = self.objects.get(instance_id)
            previous_zone_type = self.zone_type(previous["zoneId"]) if previous else None
            self.objects.setdefault(instance_id, {}).update(obj)
            new_zone_type = self.zone_type(obj.get("zoneId", -1))
            if new_zone_type in BATTLEFIELD_LIKE and previous_zone_type not in BATTLEFIELD_LIKE and turn_number is not None:
                self.entered_bf_turn[instance_id] = turn_number

        for instance_id in gsm.get("diffDeletedInstanceIds", []) or []:
            self.objects.pop(instance_id, None)

        for annotation in gsm.get("annotations", []) or []:
            if "AnnotationType_ObjectIdChanged" in annotation.get("type", []):
                orig = new = None
                for detail in annotation.get("details", []):
                    if detail.get("key") == "orig_id":
                        orig = detail["valueInt32"][0]
                    elif detail.get("key") == "new_id":
                        new = detail["valueInt32"][0]
                if orig is not None and new is not None:
                    self._id_remap[orig] = new
                    if orig in self.entered_bf_turn:
                        self.entered_bf_turn[new] = self.entered_bf_turn.pop(orig)

    def zone_owner(self, zone_id: int) -> int | None:
        zone = self.zones.get(zone_id)
        return zone.get("ownerSeatId") if zone else None

    def entered_play_this_turn(self, instance_id: int, controller_seat: int) -> bool:
        """Timing half of summoning sickness only - does NOT check haste,
        since that's a property of the card/effect, not the live game
        state this class tracks. Callers with card data should also check
        for haste before treating this as "can't attack"."""
        entered = self.entered_bf_turn.get(instance_id)
        return entered is not None and entered == self.turn_info.get("turnNumber") and self.turn_info.get("activePlayer") == controller_seat


# --------------------------------------------------------------------------
# State encoding
# --------------------------------------------------------------------------

def _resolve(grp_id: int | None, cards_by_id: dict, grp_to_index: dict[int, int], misses: Counter) -> int | None:
    """None means either "not a real card reference" or "legitimately
    hidden information" (an opponent's unrevealed card shows up as the
    placeholder "Unknown" card, not as a miss) - only a *named* card that
    fails to map to this project's card index counts as a miss."""
    if grp_id is None:
        return None
    name = cards_by_id.get(grp_id, {}).get("name")
    if name in (None, "Unknown", ""):
        return None
    index = grp_to_index.get(grp_id)
    if index is None:
        misses[name] += 1
    return index


@dataclass
class SideState:
    untapped_lands: np.ndarray
    tapped_lands: np.ndarray
    untapped_creatures: np.ndarray
    tapped_creatures: np.ndarray
    summoning_sick_creatures: np.ndarray
    noncreatures: np.ndarray
    graveyard: np.ndarray
    exile: np.ndarray
    hand: np.ndarray
    hand_size: int
    hand_known: bool
    life: float
    mana: np.ndarray  # [W, U, B, R, G, generic] untapped-land-derived, "could produce" superset


def bucket_side(
    tracker: GameStateTracker, seat: int, cards_by_id: dict, grp_to_index: dict[int, int],
    num_cards: int, misses: Counter,
) -> SideState:
    zeros = lambda: np.zeros(num_cards, dtype=np.uint8)
    untapped_lands, tapped_lands = zeros(), zeros()
    untapped_creatures, tapped_creatures, sick_creatures = zeros(), zeros(), zeros()
    noncreatures, graveyard, exile, hand = zeros(), zeros(), zeros(), zeros()
    hand_count = 0
    mana = {symbol: 0 for symbol in MANA_SYMBOLS}

    for instance_id, obj in tracker.objects.items():
        zone_type = tracker.zone_type(obj.get("zoneId", -1))
        grp_id = obj.get("grpId")
        card_types = obj.get("cardTypes", [])

        if zone_type in BATTLEFIELD_LIKE and obj.get("controllerSeatId") == seat:
            index = _resolve(grp_id, cards_by_id, grp_to_index, misses)
            tapped = bool(obj.get("isTapped"))
            if "CardType_Land" in card_types:
                bucket = tapped_lands if tapped else untapped_lands
                if index is not None:
                    bucket[index] = min(255, bucket[index] + 1)
                if not tapped:
                    for symbol in mana_colors_for(grp_id, cards_by_id):
                        mana[symbol] = mana.get(symbol, 0) + 1
            elif "CardType_Creature" in card_types:
                bucket = tapped_creatures if tapped else untapped_creatures
                if index is not None:
                    bucket[index] = min(255, bucket[index] + 1)
                    if tracker.entered_play_this_turn(instance_id, seat) and not has_keyword(grp_id, "Haste", cards_by_id):
                        sick_creatures[index] = min(255, sick_creatures[index] + 1)
            elif index is not None:
                noncreatures[index] = min(255, noncreatures[index] + 1)

        elif zone_type in HAND_LIKE and tracker.zone_owner(obj.get("zoneId", -1)) == seat:
            hand_count += 1
            index = _resolve(grp_id, cards_by_id, grp_to_index, misses)
            if index is not None:
                hand[index] = min(255, hand[index] + 1)

        elif zone_type in GRAVEYARD_LIKE and tracker.zone_owner(obj.get("zoneId", -1)) == seat:
            index = _resolve(grp_id, cards_by_id, grp_to_index, misses)
            if index is not None:
                graveyard[index] = min(255, graveyard[index] + 1)

        elif zone_type in EXILE_LIKE and tracker.zone_owner(obj.get("zoneId", -1)) == seat:
            index = _resolve(grp_id, cards_by_id, grp_to_index, misses)
            if index is not None:
                exile[index] = min(255, exile[index] + 1)

    hand_known = int(hand.sum()) >= hand_count  # every hand object resolved to a real card -> fully observed
    return SideState(
        untapped_lands=untapped_lands, tapped_lands=tapped_lands,
        untapped_creatures=untapped_creatures, tapped_creatures=tapped_creatures,
        summoning_sick_creatures=sick_creatures, noncreatures=noncreatures,
        graveyard=graveyard, exile=exile, hand=hand, hand_size=hand_count, hand_known=hand_known,
        life=float(tracker.players.get(seat, {}).get("lifeTotal", STARTING_LIFE)),
        mana=np.array([mana.get(s, 0) for s in MANA_SYMBOLS] + [0.0], dtype=np.float32),
    )


STATE_VECTOR_FIELDS = [
    "untapped_lands", "tapped_lands", "untapped_creatures", "tapped_creatures",
    "summoning_sick_creatures", "noncreatures", "graveyard", "exile", "hand",
]


def encode_state(
    tracker: GameStateTracker, deciding_seat: int, other_seat: int, decision_type: int,
    committed: np.ndarray, cards_by_id: dict, grp_to_index: dict[int, int], num_cards: int, misses: Counter,
) -> dict[str, Any]:
    """`deciding_seat` is always "self" - whichever seat the actions list
    says is choosing right now (the caller resolves this; there's no
    separate "flip" flag here because the real deciding seat is read
    directly off the data, not inferred)."""
    self_side = bucket_side(tracker, deciding_seat, cards_by_id, grp_to_index, num_cards, misses)
    opp_side = bucket_side(tracker, other_seat, cards_by_id, grp_to_index, num_cards, misses)

    record: dict[str, Any] = {
        f"self_{field}": getattr(self_side, field) for field in STATE_VECTOR_FIELDS
    }
    record.update({f"opp_{field}": getattr(opp_side, field) for field in STATE_VECTOR_FIELDS})
    record["committed_this_phase"] = committed
    record["self_life"] = self_side.life
    record["opp_life"] = opp_side.life
    record["self_hand_size"] = self_side.hand_size
    record["opp_hand_size"] = opp_side.hand_size
    record["self_hand_known"] = self_side.hand_known
    record["opp_hand_known"] = opp_side.hand_known
    record["self_available_mana"] = self_side.mana
    turn_number = tracker.turn_info.get("turnNumber") or 0
    on_play = tracker.turn_info.get("activePlayer") == deciding_seat
    record["scalars"] = np.array(
        [self_side.life, opp_side.life, float(self_side.hand_size), float(opp_side.hand_size),
         float(opp_side.hand_known), float(on_play), turn_number / 30.0, decision_type / float(PHASE_BLOCK)],
        dtype=np.float32,
    )
    return record


def flatten_state(record: dict[str, Any]) -> np.ndarray:
    vectors = [record[f"self_{f}"].astype(np.float32) for f in STATE_VECTOR_FIELDS]
    vectors += [record[f"opp_{f}"].astype(np.float32) for f in STATE_VECTOR_FIELDS]
    vectors.append(record["committed_this_phase"].astype(np.float32))
    vectors.append(record["self_available_mana"])
    vectors.append(record["scalars"])
    return np.concatenate(vectors)


# --------------------------------------------------------------------------
# Decision classification and corroboration
# --------------------------------------------------------------------------

def classify_decision_type(tracker: GameStateTracker, seat: int, instance_id: int) -> int | None:
    """Land/spell only. Combat is NOT represented as an offered-candidate
    list in this log at all (verified: `actions` never once offers a
    battlefield object across a full game) - attackers are detected
    separately in extract_decisions via the direct attackState field."""
    phase = tracker.turn_info.get("phase")
    obj = tracker.objects.get(instance_id, {})
    zone_type = tracker.zone_type(obj.get("zoneId", -1))
    card_types = obj.get("cardTypes", [])

    if phase in ("Phase_Main1", "Phase_Main2") and zone_type in HAND_LIKE:
        return PHASE_LAND if "CardType_Land" in card_types else PHASE_SPELL
    return None


def _untapped_creatures(tracker: GameStateTracker, seat: int) -> set[int]:
    candidates = set()
    for instance_id, obj in tracker.objects.items():
        if obj.get("controllerSeatId") != seat:
            continue
        if tracker.zone_type(obj.get("zoneId", -1)) not in BATTLEFIELD_LIKE:
            continue
        if "CardType_Creature" not in obj.get("cardTypes", []):
            continue
        if obj.get("isTapped"):
            continue
        candidates.add(instance_id)
    return candidates


def attacker_candidates(tracker: GameStateTracker, active_seat: int, cards_by_id: dict) -> set[int]:
    """Creatures that could plausibly be declared as attackers right now:
    untapped, no Defender. Deliberately does NOT gate on summoning sickness
    (neither the engine's own hasSummoningSickness field nor a hand-rolled
    entered-this-turn check) - hasSummoningSickness was checked directly
    against real data and found to stay True on a creature that was
    actively attacking (attackState=AttackState_Attacking), so it isn't
    trustworthy as a hard gate. Excluding on an unreliable signal risks
    silently dropping real attacks from the candidate pool before
    attackState ever gets a chance to mark them chosen, which is a worse
    failure mode than legal_mask being slightly too permissive - a creature
    that's genuinely sick will just never get attackState set and sit in
    the pool as an always-available-but-never-chosen candidate."""
    return {iid for iid in _untapped_creatures(tracker, active_seat) if not has_keyword(tracker.objects[iid].get("grpId"), "Defender", cards_by_id)}


def blocker_candidates(tracker: GameStateTracker, defending_seat: int, cards_by_id: dict) -> set[int]:
    """Creatures the defending player could declare as blockers: untapped.
    No Defender exclusion (that keyword only restricts attacking) and no
    summoning-sickness exclusion (blocking was never restricted by it in
    the first place, so there's nothing to gate on here either)."""
    return _untapped_creatures(tracker, defending_seat)


def _phase_matches(decision_type: int, phase: str | None, step: str | None) -> bool:
    if decision_type in (PHASE_LAND, PHASE_SPELL):
        return phase in ("Phase_Main1", "Phase_Main2")
    if decision_type == PHASE_ATTACK:
        return phase == "Phase_Combat" and step == "Step_DeclareAttack"
    if decision_type == PHASE_BLOCK:
        return phase == "Phase_Combat" and step == "Step_DeclareBlock"
    return False


def _corroborate(tracker: GameStateTracker, decision_type: int, instance_id: int, cards_by_id: dict) -> str:
    """Independent second signal for the land/spell elimination inference
    (attackers are directly observed via attackState, not inferred, so they
    don't need this check - see extract_decisions). A card that's chosen
    passes through ZoneType_Limbo before landing on the battlefield/stack,
    so both checks are "left hand", not "reached its final zone yet"."""
    obj = tracker.objects.get(instance_id, {})
    zone_type = tracker.zone_type(obj.get("zoneId", -1))
    if decision_type in (PHASE_LAND, PHASE_SPELL):
        return "confirmed" if zone_type not in HAND_LIKE else "contradicted"
    if decision_type in (PHASE_ATTACK, PHASE_BLOCK):
        return "directly_observed"  # attackState/blockState IS the signal, not an inference needing a check
    return "no_signal"


# --------------------------------------------------------------------------
# Main extraction loop
# --------------------------------------------------------------------------

@dataclass
class _Window:
    decision_type: int
    deciding_seat: int
    other_seat: int
    offered: set[int] = field(default_factory=set)
    committed: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.uint8))
    steps: int = 0


def _mask_for(offered: set[int], tracker: GameStateTracker, cards_by_id: dict, grp_to_index: dict[int, int], num_cards: int, misses: Counter) -> np.ndarray:
    legal = np.zeros(num_cards + 1, dtype=bool)
    legal[num_cards] = True  # STOP always legal
    for iid in offered:
        obj = tracker.objects.get(iid, {})
        index = _resolve(obj.get("grpId"), cards_by_id, grp_to_index, misses)
        if index is not None:
            legal[index] = True
    return legal


def extract_decisions(
    events: list[dict], seat_id: int, opponent_seat_id: int, won_by_seat: int | None,
    cards_by_id: dict, grp_to_index: dict[int, int], num_cards: int,
) -> tuple[list[dict], Counter, Counter]:
    tracker = GameStateTracker()
    windows: dict[tuple[int, int], _Window] = {}
    combat_windows: dict[int, _Window] = {}  # phase_id (PHASE_ATTACK/PHASE_BLOCK) -> its window, keyed separately from land/spell
    records: list[dict] = []
    diagnostics: Counter = Counter()
    misses: Counter = Counter()

    def emit(window: _Window, chosen_iid: int | None, chrono_index: int) -> None:
        """chosen_iid=None means STOP (always resolvable). A real chosen_iid
        that fails to resolve to a card index (e.g. a token with no entry in
        the project's card index) is a genuine "something was picked, we
        just can't name it" case - it must NOT fall back to the STOP action,
        since that would actively mislabel a real choice as "did nothing"
        rather than just being missing data. Such steps are skipped (not
        recorded at all) but still count toward window.steps so the
        window-close logic doesn't *also* emit a spurious closing STOP for
        a window that did have something happen in it."""
        action = num_cards
        if chosen_iid is not None:
            obj = tracker.objects.get(chosen_iid, {})
            index = _resolve(obj.get("grpId"), cards_by_id, grp_to_index, misses)
            if index is None:
                diagnostics["unresolvable_chosen_action"] += 1
                window.steps += 1
                return
            action = index
            diagnostics[_corroborate(tracker, window.decision_type, chosen_iid, cards_by_id)] += 1
            window.committed[index] = min(255, window.committed[index] + 1)

        state = encode_state(tracker, window.deciding_seat, window.other_seat, window.decision_type,
                              window.committed, cards_by_id, grp_to_index, num_cards, misses)
        legal = _mask_for(window.offered, tracker, cards_by_id, grp_to_index, num_cards, misses)
        records.append({
            "seat": window.deciding_seat,
            "phase_id": window.decision_type,
            "chrono_index": chrono_index,
            "turn_number": tracker.turn_info.get("turnNumber") or 0,
            "won": bool(won_by_seat == window.deciding_seat) if won_by_seat is not None else False,
            "state": flatten_state(state),
            "legal_mask": legal,
            "action": action,
        })
        window.steps += 1

    for chrono_index, event in enumerate(events):
        gsm = event.get("gameStateMessage")
        if not gsm:
            continue
        tracker.apply(gsm)

        if tracker.game_over:
            # Close everything cleanly (STOP, not a misattributed elimination)
            # and stop - nothing past this point is a real decision.
            for window in list(windows.values()) + list(combat_windows.values()):
                if window.offered or window.steps == 0:
                    emit(window, None, chrono_index)
            diagnostics["closed_at_game_over"] += len(windows) + len(combat_windows)
            break

        phase, step = tracker.turn_info.get("phase"), tracker.turn_info.get("step")

        # Attackers and blockers: both driven by a state field appearing on
        # a candidate (attackState / blockState), not by actions/offers -
        # combat isn't offer-driven in this log at all (see
        # classify_decision_type's docstring). The candidate pool has to be
        # computed here too, since nothing ever tells us what it is the way
        # `actions` does for land/spell. Attacking is the active player's
        # decision; blocking is the *other* player's, made during the
        # active player's turn - deciding_seat/other_seat below is that
        # distinction, and it's also why blocker_candidates takes the
        # defending seat while attacker_candidates takes the active one.
        active_seat = tracker.turn_info.get("activePlayer")
        other_of_active = (opponent_seat_id if active_seat == seat_id else seat_id) if active_seat is not None else None
        for phase_id, field_name, candidate_fn, deciding_is_active in (
            (PHASE_ATTACK, "attackState", attacker_candidates, True),
            (PHASE_BLOCK, "blockState", blocker_candidates, False),
        ):
            if _phase_matches(phase_id, phase, step) and active_seat is not None:
                deciding, other = (active_seat, other_of_active) if deciding_is_active else (other_of_active, active_seat)
                window = combat_windows.get(phase_id)
                if window is None:
                    combat_windows[phase_id] = _Window(
                        decision_type=phase_id, deciding_seat=deciding, other_seat=other,
                        offered=candidate_fn(tracker, deciding, cards_by_id),
                        committed=np.zeros(num_cards, dtype=np.uint8),
                    )
                else:
                    newly_declared = {iid for iid in window.offered if tracker.objects.get(iid, {}).get(field_name) is not None}
                    for iid in sorted(newly_declared):
                        emit(window, iid, chrono_index)
                    window.offered -= newly_declared
            else:
                window = combat_windows.pop(phase_id, None)
                if window is not None and (window.offered or window.steps == 0):
                    emit(window, None, chrono_index)

        offered_now: dict[tuple[int, int], set[int]] = {}
        raw_actions = gsm.get("actions")
        if raw_actions is not None:
            grouped: dict[int, set[int]] = {}
            for a in raw_actions:
                seat, iid = a.get("seatId"), a.get("action", {}).get("instanceId")
                if seat is None or iid is None:
                    continue
                grouped.setdefault(seat, set()).add(tracker.resolve_id(iid))
            for seat, iids in grouped.items():
                for iid in iids:
                    dtype = classify_decision_type(tracker, seat, iid)
                    if dtype is None:
                        diagnostics["unclassified_offer"] += 1
                        continue
                    offered_now.setdefault((seat, dtype), set()).add(iid)
        else:
            offered_now = None  # signal "no new info this event" below

        # close windows whose phase/step is no longer current. A window is only
        # worth a closing STOP if there was still something live to choose from
        # (offered non-empty) or nothing was ever chosen (steps == 0) - if every
        # offered candidate was already resolved via elimination, closing is
        # just administrative and would otherwise produce a redundant trailing
        # STOP after a real choice was already recorded.
        for key in list(windows.keys()):
            seat, dtype = key
            if not _phase_matches(dtype, phase, step):
                window = windows.pop(key)
                if window.offered or window.steps == 0:
                    emit(window, None, chrono_index)

        if offered_now is None:
            continue

        # Union of existing window keys and this event's offered keys - a
        # window whose candidates all vanished has no entry in offered_now at
        # all (rather than an explicit empty set), so it must still be visited
        # here (defaulted to empty) or its elimination would never be detected.
        for key in set(windows.keys()) | set(offered_now.keys()):
            seat, dtype = key
            other = opponent_seat_id if seat == seat_id else seat_id
            offered = offered_now.get(key, set())
            window = windows.get(key)
            if window is None:
                windows[key] = _Window(decision_type=dtype, deciding_seat=seat, other_seat=other,
                                        offered=offered, committed=np.zeros(num_cards, dtype=np.uint8))
                continue

            eliminated = window.offered - offered
            appeared = offered - window.offered
            if eliminated:
                if len(eliminated) > 1:
                    diagnostics["multi_elimination"] += len(eliminated)
                for iid in sorted(eliminated):
                    emit(window, iid, chrono_index)
            if appeared and not eliminated:
                diagnostics["new_candidate_appeared"] += 1
            window.offered = offered

    return records, diagnostics, misses


def find_winning_seat(events: list[dict]) -> int | None:
    winner = None
    for event in events:
        for result in event.get("gameStateMessage", {}).get("gameInfo", {}).get("results", []) or []:
            if result.get("scope") == "MatchScope_Game" and "winningTeamId" in result:
                winner = result["winningTeamId"]
    return winner


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def stack_chunk(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not records:
        return {}
    arrays: dict[str, np.ndarray] = {}
    for key in records[0]:
        values = [record[key] for record in records]
        if isinstance(values[0], np.ndarray):
            arrays[key] = np.stack(values)
        else:
            arrays[key] = np.array(values)
    return arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Turn a raw MTGA game history bundle into (state, legal_mask, action) training examples.")
    parser.add_argument("--history", type=Path, required=True, help="Path to history.json.")
    parser.add_argument("--cards", type=Path, required=True, help="Path to this bundle's cards.json.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="Card index source (card_names/card_to_index).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--game-id", default=None, help="Identifier to tag output rows with (defaults to the history.json filename stem).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.history, args.cards, args.metadata):
        if not path.exists():
            raise FileNotFoundError(path)

    data = json.loads(args.history.read_text(encoding="utf-8"))
    events, seat_id, opponent_seat_id = data["events"], data["seat_id"], data["opponent_seat_id"]
    game_id = args.game_id or args.history.stem

    card_names, card_to_index = load_card_index(args.metadata)
    num_cards = len(card_names)
    cards_by_id = load_cards_bundle(args.cards)
    grp_to_index, grp_seen, grp_resolved = build_grp_id_index(cards_by_id, card_to_index)
    print(f"card resolution: {grp_resolved}/{grp_seen} named cards in this bundle mapped to the project card index")

    won_by_seat = find_winning_seat(events)
    print(f"seat_id={seat_id} opponent_seat_id={opponent_seat_id} won_by_seat={won_by_seat}")

    records, diagnostics, misses = extract_decisions(events, seat_id, opponent_seat_id, won_by_seat, cards_by_id, grp_to_index, num_cards)
    print(f"extracted {len(records):,} decision steps from {len(events):,} events")
    print(f"by phase: { {PHASE_NAMES[p]: sum(1 for r in records if r['phase_id'] == p) for p in PHASE_NAMES} }")
    signal_keys = ("confirmed", "contradicted", "directly_observed", "no_signal")
    print(f"action-inference signal: { {k: v for k, v in diagnostics.items() if k in signal_keys} }")
    print(f"other diagnostics: { {k: v for k, v in diagnostics.items() if k not in signal_keys} }")
    if misses:
        print(f"unresolved card names (top 10): {misses.most_common(10)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        record["game_id"] = game_id
    arrays = stack_chunk(records)
    out_path = args.output_dir / f"history_steps_{game_id}.npz"
    if arrays:
        np.savez_compressed(out_path, **arrays)
        print(f"wrote {out_path}")

    metadata = {
        "source_history": str(args.history),
        "source_cards": str(args.cards),
        "game_id": game_id,
        "seat_id": seat_id,
        "opponent_seat_id": opponent_seat_id,
        "won_by_seat": won_by_seat,
        "num_cards": num_cards,
        "action_space": num_cards + 1,
        "stop_action_index": num_cards,
        "phase_ids": {name: pid for pid, name in PHASE_NAMES.items()},
        "state_layout": (
            "concat[self_{untapped_lands,tapped_lands,untapped_creatures,tapped_creatures,"
            "summoning_sick_creatures,noncreatures,graveyard,exile,hand}(num_cards each), "
            "opp_{...same 9...}(num_cards each), committed_this_phase(num_cards), "
            "self_available_mana(6: W,U,B,R,G,generic), "
            "scalars(self_life,opp_life,self_hand_size,opp_hand_size,opp_hand_known,on_play,"
            "turn_number/30,phase/5)]"
        ),
        "state_dim": num_cards * 19 + 6 + 8,  # 9 self + 9 opp + 1 committed_this_phase count-vectors, + mana(6) + scalars(8)
        "steps_written": len(records),
        "events_processed": len(events),
        "grp_id_ids_seen": grp_seen,
        "grp_id_ids_resolved": grp_resolved,
        "diagnostics": dict(diagnostics),
        "unresolved_card_name_hits": dict(misses.most_common(50)),
        "notes": (
            "Land/spell/attack/block. Attackers and blockers are both read directly off the "
            "engine's own state fields (attackState / blockState), not inferred - confirmed "
            "against a real block: the blocking creature gets blockState "
            "Declared->Blocking plus blockInfo.attackerIds, and the attacker's own object "
            "gets blockState=Blocked at the same event. Neither is gated on "
            "hasSummoningSickness (checked directly against real data and found to stay True "
            "on a creature that was actively attacking, so it isn't trustworthy) - the legal "
            "candidate pool is just untapped (+ no Defender, for attackers only), and the "
            "observed state field is the sole arbiter of what was actually chosen. Targeting "
            "and trigger/effect ordering (ClientMessageType_SelectNResp) are detected but not "
            "yet extracted into rows; mulligan decisions are out of scope. "
            "`diagnostics.confirmed/contradicted` is the corroboration check on the land/spell "
            "elimination inference; `directly_observed` marks attacker/blocker steps, which "
            "don't need corroboration since attackState/blockState is ground truth, not an "
            "inference."
        ),
    }
    with (args.output_dir / f"history_steps_{game_id}.metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
