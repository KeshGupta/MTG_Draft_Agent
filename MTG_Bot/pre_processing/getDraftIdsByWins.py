import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = ROOT / "SL" / "data" / "raw" / "draft_data_public.SOS.PremierDraft.csv"


def matching_draft_ids(input_csv: Path, wins: int) -> list[str]:
    draft_ids: list[str] = []
    seen: set[str] = set()

    with input_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        draft_id_index = header.index("draft_id")
        wins_index = header.index("event_match_wins")

        for row in reader:
            if int(row[wins_index]) != wins:
                continue

            draft_id = row[draft_id_index]
            if draft_id in seen:
                continue

            seen.add(draft_id)
            draft_ids.append(draft_id)

    return draft_ids


def save_draft_ids(output_path: Path, draft_ids: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for draft_id in draft_ids:
            file.write(f"{draft_id}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save draft IDs from a 17Lands draft CSV that finished with a specific number of wins."
    )
    parser.add_argument("--wins", type=int, required=True, help="Exact event_match_wins value to keep.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="17Lands draft CSV path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output txt path. Defaults beside the input CSV as draft_ids_<wins>_wins.txt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output or args.input_csv.with_name(f"draft_ids_{args.wins}_wins.txt")
    draft_ids = matching_draft_ids(args.input_csv, args.wins)
    save_draft_ids(output_path, draft_ids)
    print(f"Saved {len(draft_ids)} draft IDs with {args.wins} wins to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
