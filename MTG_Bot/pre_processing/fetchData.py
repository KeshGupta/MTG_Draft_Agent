import argparse
from email.utils import parsedate_to_datetime
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Iterator

import requests


BASE = "https://www.17lands.com/data"
DEFAULT_DRAFT_ID = "5f2f3d857cae4325bcea3290f4d3964a"
DEFAULT_DRAFT_IDS = (DEFAULT_DRAFT_ID,)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "raw_replays"
CARD_ID_FIELDS = ("grpId", "overlayGrpId", "objectSourceGrpId")
JSONL_FILENAMES = {
    "history_info": "history_info.jsonl",
    "history": "history.jsonl",
    "manifest": "manifest.jsonl",
}
UNIVERSAL_CARDS_FILENAME = "cards.json"
APPEND_TRANSACTION_FILENAME = ".append_game_row.json"
DEFAULT_MAX_MATCHES_TO_SCAN = 20
DEFAULT_MAX_GAMES_TO_SCAN = 3
DEFAULT_REQUEST_DELAY_SECONDS = 1.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 8
DEFAULT_BACKOFF_SECONDS = 5.0
DEFAULT_MAX_BACKOFF_SECONDS = 300.0
RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}

request_delay_seconds = DEFAULT_REQUEST_DELAY_SECONDS
request_timeout_seconds = DEFAULT_REQUEST_TIMEOUT_SECONDS
max_retries = DEFAULT_MAX_RETRIES
backoff_seconds = DEFAULT_BACKOFF_SECONDS
max_backoff_seconds = DEFAULT_MAX_BACKOFF_SECONDS
last_request_at = 0.0

session = requests.Session()


def configure_request_settings(
    request_delay: float,
    request_timeout: float,
    retries: int,
    backoff: float,
    max_backoff: float,
) -> None:
    global request_delay_seconds
    global request_timeout_seconds
    global max_retries
    global backoff_seconds
    global max_backoff_seconds

    request_delay_seconds = request_delay
    request_timeout_seconds = request_timeout
    max_retries = retries
    backoff_seconds = backoff
    max_backoff_seconds = max_backoff


def wait_for_request_slot() -> None:
    global last_request_at

    if request_delay_seconds <= 0:
        last_request_at = time.monotonic()
        return

    elapsed = time.monotonic() - last_request_at
    if elapsed < request_delay_seconds:
        time.sleep(request_delay_seconds - elapsed)

    last_request_at = time.monotonic()


def retry_after_seconds(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None

    try:
        delay = float(value)
        if delay >= 0:
            return delay
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.astimezone()

    return max(0.0, retry_at.timestamp() - time.time())


def retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = retry_after_seconds(response)
        if retry_after is not None:
            return min(retry_after, max_backoff_seconds)

    exponential_delay = backoff_seconds * (2 ** (attempt - 1))
    jitter = random.uniform(0, min(1.0, backoff_seconds))
    return min(exponential_delay + jitter, max_backoff_seconds)


def should_retry_response(response: requests.Response) -> bool:
    return response.status_code in RETRYABLE_STATUS_CODES


def get_json(endpoint: str, params: dict[str, Any]) -> Any:
    url = f"{BASE}/{endpoint}"
    last_error: requests.RequestException | None = None

    for attempt in range(1, max_retries + 2):
        wait_for_request_slot()
        response: requests.Response | None = None

        try:
            response = session.get(url, params=params, timeout=request_timeout_seconds)
            if should_retry_response(response) and attempt <= max_retries:
                delay = retry_delay(response, attempt)
                print(
                    f"{response.status_code} from 17Lands for {endpoint}; "
                    f"retrying in {delay:.1f}s ({attempt}/{max_retries})",
                    flush=True,
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            last_error = error
            if attempt > max_retries:
                raise

            delay = retry_delay(response, attempt)
            print(
                f"Request failed for {endpoint}: {error}; "
                f"retrying in {delay:.1f}s ({attempt}/{max_retries})",
                flush=True,
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to fetch {endpoint}")



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
    unique_ids = sorted(set(card_ids))
    if not unique_ids:
        return {"cards": [], "emblems": []}
    ids = ",".join(str(card_id) for card_id in unique_ids)
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


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temp_path.replace(path)


def jsonl_line(data: Any) -> bytes:
    return (json.dumps(data, separators=(",", ":")) + "\n").encode("utf-8")


def file_size(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_size


def truncate_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as file:
        file.truncate(size)
        file.flush()
        os.fsync(file.fileno())


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def jsonl_paths(output_dir: Path) -> dict[str, Path]:
    return {name: output_dir / filename for name, filename in JSONL_FILENAMES.items()}


def append_transaction_path(output_dir: Path) -> Path:
    return output_dir / APPEND_TRANSACTION_FILENAME


def rollback_pending_append(output_dir: Path) -> None:
    transaction_path = append_transaction_path(output_dir)
    if not transaction_path.exists():
        return

    transaction = load_json(transaction_path, {})
    byte_offsets = transaction.get("byte_offsets")
    files = transaction.get("files")
    if not isinstance(byte_offsets, dict) or not isinstance(files, dict):
        raise ValueError(f"Cannot recover malformed append transaction at {transaction_path}")

    for name, filename in files.items():
        if name not in JSONL_FILENAMES:
            raise ValueError(f"Cannot recover unknown JSONL file {name!r} in {transaction_path}")
        truncate_file(output_dir / str(filename), int(byte_offsets[name]))

    transaction_path.unlink()


def next_jsonl_row_index(output_dir: Path) -> int:
    rollback_pending_append(output_dir)
    paths = jsonl_paths(output_dir)
    row_counts = {name: count_jsonl_rows(path) for name, path in paths.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(
            "JSONL files are not aligned: "
            + ", ".join(f"{name}={count}" for name, count in row_counts.items())
        )
    return next(iter(row_counts.values()))


def load_existing_game_keys(output_dir: Path) -> set[tuple[str, int, int]]:
    path = output_dir / JSONL_FILENAMES["manifest"]
    existing: set[tuple[str, int, int]] = set()
    if not path.exists():
        return existing

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            manifest = json.loads(line)
            try:
                existing.add(
                    (
                        str(manifest["draft_id"]),
                        int(manifest["match_index"]),
                        int(manifest["game_index"]),
                    )
                )
            except KeyError as error:
                raise ValueError(f"Manifest row {line_number} is missing {error.args[0]!r}") from error

    return existing


def object_key(value: Any) -> tuple[str, str]:
    if isinstance(value, dict) and "id" in value:
        return ("id", str(value["id"]))
    return ("json", json.dumps(value, sort_keys=True, separators=(",", ":")))


def sort_keyed_values(values: dict[tuple[str, str], Any]) -> list[Any]:
    return [values[key] for key in sorted(values)]


def merge_universal_cards(cards_path: Path, fetched_cards: dict[str, Any]) -> list[int]:
    cards_data = load_json(cards_path, {"cards": [], "emblems": []})
    cards_data.setdefault("cards", [])
    cards_data.setdefault("emblems", [])

    cards_by_key = {object_key(card): card for card in cards_data["cards"]}
    emblems_by_key = {object_key(emblem): emblem for emblem in cards_data["emblems"]}
    added_card_ids: list[int] = []

    for card in fetched_cards.get("cards", []):
        key = object_key(card)
        if key not in cards_by_key:
            cards_by_key[key] = card
            if isinstance(card, dict) and isinstance(card.get("id"), int):
                added_card_ids.append(card["id"])

    for emblem in fetched_cards.get("emblems", []):
        key = object_key(emblem)
        if key not in emblems_by_key:
            emblems_by_key[key] = emblem

    cards_data["cards"] = sort_keyed_values(cards_by_key)
    cards_data["emblems"] = sort_keyed_values(emblems_by_key)
    save_json(cards_path, cards_data)
    return sorted(added_card_ids)


def known_card_ids(cards_path: Path) -> set[int]:
    cards_data = load_json(cards_path, {"cards": []})
    ids: set[int] = set()
    for card in cards_data.get("cards", []):
        if isinstance(card, dict):
            add_int(card.get("id"), ids)
    return ids


def update_universal_cards(output_dir: Path, card_ids: Iterable[int]) -> list[int]:
    cards_path = output_dir / UNIVERSAL_CARDS_FILENAME
    missing_card_ids = sorted(set(card_ids) - known_card_ids(cards_path))
    fetched_cards = get_cards(missing_card_ids)
    return merge_universal_cards(cards_path, fetched_cards)


def append_game_jsonl(
    output_dir: Path,
    history_info: dict[str, Any],
    history: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    rollback_pending_append(output_dir)
    paths = jsonl_paths(output_dir)
    expected_row_index = next_jsonl_row_index(output_dir)
    if manifest["row_index"] != expected_row_index:
        raise ValueError(
            f"Manifest row_index {manifest['row_index']} does not match next JSONL row {expected_row_index}"
        )

    lines = {
        "history_info": jsonl_line(history_info),
        "history": jsonl_line(history),
        "manifest": jsonl_line(manifest),
    }
    transaction = {
        "row_index": manifest["row_index"],
        "files": JSONL_FILENAMES,
        "byte_offsets": {name: file_size(path) for name, path in paths.items()},
    }
    transaction_path = append_transaction_path(output_dir)
    save_json_atomic(transaction_path, transaction)

    try:
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as file:
                file.write(lines[name])
                file.flush()
                os.fsync(file.fileno())
    except Exception:
        rollback_pending_append(output_dir)
        raise

    transaction_path.unlink()


def get_history_info_or_none(draft_id: str, match_index: int, game_index: int) -> dict[str, Any] | None:
    try:
        info = get_history_info(draft_id, match_index, game_index)
    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code in {400, 404}:
            return None
        raise

    if not isinstance(info, dict) or "history_path" not in info:
        return None
    return info


def int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def event_info_value(history_info: dict[str, Any], key: str) -> int | None:
    event_info = history_info.get("event_info")
    if not isinstance(event_info, dict):
        return None
    return int_or_none(event_info.get(key))


def discovered_match_indices(history_info: dict[str, Any], max_match_index: int | None) -> range:
    if max_match_index is not None:
        return range(max_match_index + 1)

    wins = event_info_value(history_info, "wins")
    losses = event_info_value(history_info, "losses")
    if wins is not None and losses is not None and wins + losses > 0:
        return range(wins + losses)

    return range(DEFAULT_MAX_MATCHES_TO_SCAN)


def discovered_game_indices(history_info: dict[str, Any], max_game_index: int | None) -> range:
    if max_game_index is not None:
        return range(max_game_index + 1)

    best_of_n = event_info_value(history_info, "best_of_n")
    if best_of_n is not None and best_of_n > 0:
        return range(best_of_n)

    return range(DEFAULT_MAX_GAMES_TO_SCAN)


def iter_history_infos(
    draft_id: str,
    match_index: int | None = None,
    game_index: int | None = None,
    max_match_index: int | None = None,
    max_game_index: int | None = None,
    existing_game_keys: set[tuple[str, int, int]] | None = None,
    append_existing: bool = False,
) -> Iterator[tuple[int, int, dict[str, Any]]]:
    if match_index is not None and game_index is not None:
        if (
            not append_existing
            and existing_game_keys is not None
            and (draft_id, match_index, game_index) in existing_game_keys
        ):
            return
        yield match_index, game_index, get_history_info(draft_id, match_index, game_index)
        return

    seed_match_index = match_index if match_index is not None else 0
    seed_info = get_history_info(draft_id, seed_match_index, 0)

    match_indices: Iterable[int]
    if match_index is None:
        match_indices = discovered_match_indices(seed_info, max_match_index)
    else:
        match_indices = (match_index,)

    game_indices: Iterable[int]
    if game_index is None:
        game_indices = discovered_game_indices(seed_info, max_game_index)
    else:
        game_indices = (game_index,)

    for current_match_index in match_indices:
        for current_game_index in game_indices:
            game_key = (draft_id, current_match_index, current_game_index)
            if not append_existing and existing_game_keys is not None and game_key in existing_game_keys:
                continue

            if current_match_index == seed_match_index and current_game_index == 0:
                info = seed_info
            else:
                info = get_history_info_or_none(draft_id, current_match_index, current_game_index)

            if info is not None:
                yield current_match_index, current_game_index, info


def fetch_replay_bundle(
    draft_id: str,
    match_index: int,
    game_index: int,
    output_dir: Path,
    row_index: int | None = None,
    append_existing: bool = False,
    existing_game_keys: set[tuple[str, int, int]] | None = None,
) -> dict[str, Any] | None:
    info = get_history_info(draft_id, match_index, game_index)
    return fetch_and_store_game(
        draft_id,
        match_index,
        game_index,
        info,
        output_dir,
        row_index,
        append_existing,
        existing_game_keys,
    )


def fetch_and_store_game(
    draft_id: str,
    match_index: int,
    game_index: int,
    info: dict[str, Any],
    output_dir: Path,
    row_index: int | None = None,
    append_existing: bool = False,
    existing_game_keys: set[tuple[str, int, int]] | None = None,
) -> dict[str, Any] | None:
    if existing_game_keys is None:
        existing_game_keys = load_existing_game_keys(output_dir)

    game_key = (draft_id, match_index, game_index)
    if not append_existing and game_key in existing_game_keys:
        return None

    history_path = info["history_path"]
    history = get_history(history_path)
    card_ids = collect_card_ids(history)
    new_card_ids_added = update_universal_cards(output_dir, card_ids)

    if row_index is None:
        row_index = next_jsonl_row_index(output_dir)

    manifest = {
        "draft_id": draft_id,
        "match_index": match_index,
        "game_index": game_index,
        "row_index": row_index,
        "history_path": history_path,
        "card_ids_requested": card_ids,
        "new_card_ids_added": new_card_ids_added,
        "files": {
            "history_info": JSONL_FILENAMES["history_info"],
            "history": JSONL_FILENAMES["history"],
            "manifest": JSONL_FILENAMES["manifest"],
            "cards": UNIVERSAL_CARDS_FILENAME,
        },
    }

    append_game_jsonl(output_dir, info, history, manifest)
    existing_game_keys.add(game_key)
    return manifest


def fetch_drafts(
    draft_ids: Iterable[str],
    output_dir: Path,
    match_index: int | None = None,
    game_index: int | None = None,
    max_match_index: int | None = None,
    max_game_index: int | None = None,
    append_existing: bool = False,
) -> list[dict[str, Any]]:
    rollback_pending_append(output_dir)
    existing_game_keys = set() if append_existing else load_existing_game_keys(output_dir)
    stored_manifests: list[dict[str, Any]] = []

    for draft_id in unique_ordered(draft_ids):
        for current_match_index, current_game_index, info in iter_history_infos(
            draft_id,
            match_index=match_index,
            game_index=game_index,
            max_match_index=max_match_index,
            max_game_index=max_game_index,
            existing_game_keys=existing_game_keys,
            append_existing=append_existing,
        ):
            row_index = next_jsonl_row_index(output_dir)
            manifest = fetch_and_store_game(
                draft_id,
                current_match_index,
                current_game_index,
                info,
                output_dir,
                row_index=row_index,
                append_existing=append_existing,
                existing_game_keys=existing_game_keys,
            )
            if manifest is not None:
                stored_manifests.append(manifest)

    return stored_manifests


def unique_ordered(values: Iterable[str]) -> list[str]:
    unique_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def expand_draft_ids(values: Iterable[str]) -> list[str]:
    draft_ids: list[str] = []
    for value in values:
        text = value.strip()
        if not text:
            continue
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("--draft-ids JSON value must be an array")
            draft_ids.extend(str(item).strip() for item in parsed if str(item).strip())
            continue
        draft_ids.extend(part.strip() for part in text.split(",") if part.strip())
    return draft_ids


def read_draft_id_file(path: Path) -> list[str]:
    draft_ids: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            draft_ids.extend(expand_draft_ids([text]))
    return draft_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch 17Lands replay data into aligned JSONL files and one universal cards.json."
    )
    parser.add_argument(
        "--draft-id",
        action="append",
        dest="draft_ids",
        help="Draft ID to fetch. Repeat this option for multiple draft IDs.",
    )
    parser.add_argument(
        "--draft-ids",
        nargs="+",
        dest="draft_ids_many",
        help="Draft IDs to fetch. Accepts space-separated IDs, comma-separated IDs, or a JSON array string.",
    )
    parser.add_argument(
        "--draft-id-file",
        action="append",
        type=Path,
        default=[],
        help="Txt file of draft IDs to fetch. One ID per line; blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--match-index",
        type=int,
        default=None,
        help="Fetch only this match index. Defaults to all matches discovered for each draft.",
    )
    parser.add_argument(
        "--game-index",
        type=int,
        default=None,
        help="Fetch only this game index. Defaults to all games discovered for each match.",
    )
    parser.add_argument(
        "--max-match-index",
        type=int,
        default=None,
        help="When match count cannot be discovered, scan through this match index.",
    )
    parser.add_argument(
        "--max-game-index",
        type=int,
        default=None,
        help="When game count cannot be discovered, scan through this game index.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Append rows even if a draft/match/game tuple is already present in manifest.jsonl.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
        help="Minimum seconds to wait between 17Lands requests.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Seconds to wait before an individual 17Lands request times out.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retries for 403, 429, 5xx, and temporary request failures.",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=DEFAULT_BACKOFF_SECONDS,
        help="Initial exponential backoff delay in seconds.",
    )
    parser.add_argument(
        "--max-backoff",
        type=float,
        default=DEFAULT_MAX_BACKOFF_SECONDS,
        help="Maximum retry backoff delay in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_request_settings(
        args.request_delay,
        args.request_timeout,
        args.max_retries,
        args.backoff,
        args.max_backoff,
    )

    draft_id_args = (args.draft_ids or []) + (args.draft_ids_many or [])
    draft_ids = expand_draft_ids(draft_id_args)
    for draft_id_file in args.draft_id_file:
        draft_ids.extend(read_draft_id_file(draft_id_file))

    if not draft_ids:
        draft_ids = list(DEFAULT_DRAFT_IDS)

    manifests = fetch_drafts(
        draft_ids,
        args.output_dir,
        match_index=args.match_index,
        game_index=args.game_index,
        max_match_index=args.max_match_index,
        max_game_index=args.max_game_index,
        append_existing=args.append_existing,
    )

    print(f"Saved {len(manifests)} game rows to {args.output_dir}")
    print(f"History info: {args.output_dir / JSONL_FILENAMES['history_info']}")
    print(f"History: {args.output_dir / JSONL_FILENAMES['history']}")
    print(f"Manifest: {args.output_dir / JSONL_FILENAMES['manifest']}")
    print(f"Cards: {args.output_dir / UNIVERSAL_CARDS_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
