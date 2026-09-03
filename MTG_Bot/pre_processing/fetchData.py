import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import requests


BASE = "https://www.17lands.com/data"
DEFAULT_DRAFT_ID = "5f2f3d857cae4325bcea3290f4d3964a"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "raw_replays"
CARD_ID_FIELDS = ("grpId", "overlayGrpId", "objectSourceGrpId")

session = requests.Session()


def get_json(endpoint: str, params: dict[str, Any]) -> Any:
    response = session.get(f"{BASE}/{endpoint}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_history_info(draft_id: str, match_index: int, game_index: int = 0) -> dict[str, Any]:
    return get_json(
        "history_info/",
        {
            "draft_id": draft_id,
            "match_index": match_index,
            "game_index": game_index,
        },
    )


def get_history(history_path: str) -> dict[str, Any]:
    return get_json("history", {"history_path": history_path})


def get_cards(card_ids: Iterable[int]) -> dict[str, Any]:
    ids = ",".join(str(card_id) for card_id in sorted(set(card_ids)))
    return get_json("cards", {"ids": ids})


def add_int(value: Any, card_ids: set[int]) -> None:
    if isinstance(value, int):
        card_ids.add(value)
    elif isinstance(value, str) and value.isdigit():
        card_ids.add(int(value))


def collect_card_ids(history: dict[str, Any]) -> list[int]:
    """Collect stable card-ish IDs needed to decode replay object instance IDs."""
    card_ids: set[int] = set()

    for event in history.get("events", []):
        message = event.get("gameStateMessage") or {}
        for obj in message.get("gameObjects", []):
            for field in CARD_ID_FIELDS:
                add_int(obj.get(field), card_ids)

    return sorted(card_ids)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def output_folder(output_dir: Path, draft_id: str, match_index: int, game_index: int) -> Path:
    return output_dir / draft_id / f"match_{match_index}_game_{game_index}"


def fetch_replay_bundle(draft_id: str, match_index: int, game_index: int, output_dir: Path) -> Path:
    info = get_history_info(draft_id, match_index, game_index)
    history_path = info["history_path"]
    history = get_history(history_path)
    card_ids = collect_card_ids(history)
    cards = get_cards(card_ids)

    folder = output_folder(output_dir, draft_id, match_index, game_index)
    manifest = {
        "draft_id": draft_id,
        "match_index": match_index,
        "game_index": game_index,
        "history_path": history_path,
        "card_ids_requested": card_ids,
        "files": {
            "history_info": "history_info.json",
            "history": "history.json",
            "cards": "cards.json",
        },
    }

    save_json(folder / "history_info.json", info)
    save_json(folder / "history.json", history)
    save_json(folder / "cards.json", cards)
    save_json(folder / "manifest.json", manifest)

    return folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch every 17Lands file needed to decode one Arena replay into state/action pairs."
    )
    parser.add_argument("--draft-id", default=DEFAULT_DRAFT_ID)
    parser.add_argument("--match-index", type=int, default=0)
    parser.add_argument("--game-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder = fetch_replay_bundle(args.draft_id, args.match_index, args.game_index, args.output_dir)
    print(f"Saved replay bundle to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
