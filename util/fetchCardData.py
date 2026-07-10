#!/usr/bin/env python3
"""Fetch every Scryfall card object for a Magic set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SCRYFALL_SEARCH_URL = "https://api.scryfall.com/cards/search"
REQUEST_DELAY_SECONDS = 0.11

HEADERS = {
    "User-Agent": "MTG_Draft_Agent/0.1 (local script)",
    "Accept": "application/json;q=0.9,*/*;q=0.8",
}


class ScryfallError(RuntimeError):
    """Raised when Scryfall returns an error response."""


def request_json(url: str) -> dict[str, Any]:
    request = Request(url, headers=HEADERS)

    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.reason
        try:
            body = json.loads(error.read().decode("utf-8"))
            detail = body.get("details") or body.get("warning") or detail
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        raise ScryfallError(f"Scryfall request failed ({error.code}): {detail}") from error
    except URLError as error:
        raise ScryfallError(f"Could not reach Scryfall: {error.reason}") from error


def build_search_url(
    set_code: str,
    include_extras: bool,
    include_multilingual: bool,
) -> str:
    params = {
        "q": f"e:{set_code}",
        "unique": "prints",
        "order": "set",
        "include_extras": str(include_extras).lower(),
        "include_multilingual": str(include_multilingual).lower(),
    }
    return f"{SCRYFALL_SEARCH_URL}?{urlencode(params)}"


def fetch_cards_for_set(
    set_code: str,
    include_extras: bool = False,
    include_multilingual: bool = False,
) -> list[dict[str, Any]]:
    """Return all paginated card records for a set code."""

    cards: list[dict[str, Any]] = []
    next_url: str | None = build_search_url(
        set_code=set_code.lower(),
        include_extras=include_extras,
        include_multilingual=include_multilingual,
    )

    while next_url:
        page = request_json(next_url)
        cards.extend(page.get("data", []))

        if page.get("has_more"):
            next_url = page.get("next_page")
            time.sleep(REQUEST_DELAY_SECONDS)
        else:
            next_url = None

    return cards


def write_cards(cards: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cards, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch every Scryfall card object printed in a Magic set.",
    )
    parser.add_argument(
        "set_code",
        help="Scryfall set code, such as dft, blb, mh3, or otj.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="JSON output path. Defaults to data/<set_code>_cards.json.",
    )
    parser.add_argument(
        "--include-extras",
        action="store_true",
        help="Include extra cards such as tokens, art cards, and promos.",
    )
    parser.add_argument(
        "--include-multilingual",
        action="store_true",
        help="Include non-English printings.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_code = "args.set_code.strip().lower()"

    if not set_code:
        print("Set code cannot be empty.", file=sys.stderr)
        return 2

    output_path = args.output or Path("data") / f"{set_code}_cards.json"

    try:
        cards = fetch_cards_for_set(
            set_code,
            include_extras=args.include_extras,
            include_multilingual=args.include_multilingual,
        )
    except ScryfallError as error:
        print(error, file=sys.stderr)
        return 1

    write_cards(cards, output_path)
    print(f"Fetched {len(cards)} cards from {set_code.upper()} into {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
