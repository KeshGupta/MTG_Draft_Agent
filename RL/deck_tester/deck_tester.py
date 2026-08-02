import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random
import shutil
import time

class deck_tester:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.test_dir = self.base_dir / "test"
        self.pool_dir = self.base_dir / "pool"
        self.java_exe = Path(r"C:\Users\samth\.jdks\ms-17.0.19\bin\java.exe")
        self.jar_file = self.base_dir / "forge-gui-desktop-2.0.14-SNAPSHOT-jar-with-dependencies.jar"
        self.working_dir = Path(r"C:\Users\samth\Downloads\forge\forge-gui")

    def test_batch(self, decks, num_games=5, best_of=1, timeout=300, seed=None, workers=4, return_details=False):
        if not decks:
            return []

        workers = max(1, workers)
        jobs = self._make_jobs(decks, num_games, workers)
        totals = [
            {"matches": 0, "wins": 0, "losses": 0, "ties": 0}
            for _ in decks
        ]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for job in jobs:
                futures.append(executor.submit(
                    self._run_deck_job,
                    job,
                    best_of,
                    timeout,
                    seed,
                ))

            for future in as_completed(futures):
                for deck_index, stats in future.result():
                    totals[deck_index]["matches"] += stats["matches"]
                    totals[deck_index]["wins"] += stats["wins"]
                    totals[deck_index]["losses"] += stats["losses"]
                    totals[deck_index]["ties"] += stats["ties"]

        for deck_index, total in enumerate(totals):
            if total["matches"] != num_games:
                raise RuntimeError(
                    f"Expected {num_games} matches for deck_{deck_index:04d}, "
                    f"Forge reported {total['matches']}."
                )

        win_rates = [
            total["wins"] * 100.0 / total["matches"] if total["matches"] else 0.0
            for total in totals
        ]
        if not return_details:
            return win_rates

        return {
            "win_rates": win_rates,
            "per_deck": [
                {
                    "deck_index": deck_index,
                    "matches": total["matches"],
                    "expected_matches": num_games,
                    "wins": total["wins"],
                    "losses": total["losses"],
                    "ties": total["ties"],
                    "win_rate": win_rates[deck_index] / 100.0,
                }
                for deck_index, total in enumerate(totals)
            ],
        }

    def test_batch_against_pool(
        self,
        decks,
        opponents_per_deck=4,
        games_per_opponent=20,
        best_of=1,
        timeout=300,
        seed=None,
        workers=8,
        return_details=False,
    ):
        """Test each deck against the same sampled pool decks.

        Returns one win-rate percentage per submitted deck. Each generated deck
        is run against the same ``opponents_per_deck`` opponent .dck files for
        ``games_per_opponent`` games each.
        """
        if not decks:
            return []
        if opponents_per_deck <= 0:
            raise ValueError("opponents_per_deck must be positive.")
        if games_per_opponent <= 0:
            raise ValueError("games_per_opponent must be positive.")
        if best_of != 1:
            raise ValueError("Raw game-count rewards require best_of=1.")

        opponent_decks = sorted(self.pool_dir.glob("*.dck"))
        if len(opponent_decks) < opponents_per_deck:
            raise ValueError(
                f"Need at least {opponents_per_deck} opponent decks in {self.pool_dir}, "
                f"found {len(opponent_decks)}."
            )

        rng = random.Random(seed)
        sampled_opponents = rng.sample(opponent_decks, opponents_per_deck)
        jobs = []
        for deck_index, deck in enumerate(decks):
            for opponent_index, opponent_path in enumerate(sampled_opponents):
                jobs.append({
                    "job_id": len(jobs),
                    "deck_index": deck_index,
                    "deck": deck,
                    "deck_name": f"deck_{deck_index:04d}_opp_{opponent_index:02d}",
                    "opponent_index": opponent_index,
                    "opponent_path": opponent_path,
                    "num_games": games_per_opponent,
                })

        totals = [
            {"games": 0, "wins": 0, "losses": 0, "ties": 0}
            for _ in decks
        ]
        matchup_details = [
            []
            for _ in decks
        ]

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [
                executor.submit(self._run_matchup_job, job, best_of, timeout, seed)
                for job in jobs
            ]
            for future in as_completed(futures):
                deck_index, stats = future.result()
                totals[deck_index]["games"] += stats["games"]
                totals[deck_index]["wins"] += stats["wins"]
                totals[deck_index]["losses"] += stats["losses"]
                totals[deck_index]["ties"] += stats["ties"]
                matchup_details[deck_index].append(stats)

        win_rates = [
            total["wins"] * 100.0 / total["games"] if total["games"] else 0.0
            for total in totals
        ]
        if not return_details:
            return win_rates

        return {
            "win_rates": win_rates,
            "opponents": [path.name for path in sampled_opponents],
            "per_deck": [
                {
                    "deck_index": deck_index,
                    "games": total["games"],
                    "wins": total["wins"],
                    "losses": total["losses"],
                    "ties": total["ties"],
                    "win_rate": win_rates[deck_index] / 100.0,
                    "matchups": sorted(matchup_details[deck_index], key=lambda row: row["opponent_index"]),
                }
                for deck_index, total in enumerate(totals)
            ],
        }

    def _run_deck_job(self, job, best_of, timeout, seed):
        worker_test_dir = self.base_dir / "worker_tests" / f"job_{job['job_id']:03d}"
        self.clear_deck_dir(worker_test_dir)

        deck_names = []
        for deck_index, deck, deck_name in job["decks"]:
            deck_names.append((deck_index, deck_name))
            self.write_deck(worker_test_dir, deck_name, deck)

        command = [
            str(self.java_exe),
            "-Xmx1536m",
            "-jar",
            str(self.jar_file),
            "sim",
            "-t",
            "DeckTest",
            "-testDir",
            str(worker_test_dir),
            "-poolDir",
            str(self.pool_dir),
            "-s",
            str(job["num_games"]),
            "-m",
            str(best_of),
            "-q",
            "-c",
            str(timeout),
        ]

        if seed is not None:
            command += ["-seed", str(seed + job["job_id"])]

        result = subprocess.run(
            command,
            cwd=self.working_dir,
            capture_output=True,
            text=True,
            creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Deck test failed with exit code {result.returncode}\n"
                f"STDOUT:\n{result.stdout[-2000:]}\n"
                f"STDERR:\n{result.stderr[-2000:]}"
            )

        results_by_deck = self._parse_deck_results(result.stdout)
        missing = [deck_name for _, deck_name in deck_names if deck_name not in results_by_deck]
        if missing:
            raise RuntimeError(
                f"Forge output did not include results for {missing}. "
                f"Parsed decks: {sorted(results_by_deck)}\n"
                f"STDOUT:\n{result.stdout[-2000:]}"
            )
        return [
            (deck_index, results_by_deck[deck_name])
            for deck_index, deck_name in deck_names
        ]

    def _run_matchup_job(self, job, best_of, timeout, seed):
        worker_dir = self.base_dir / "worker_tests" / "matchups" / f"job_{job['job_id']:04d}"
        worker_test_dir = worker_dir / "test"
        worker_pool_dir = worker_dir / "pool"
        self.clear_deck_dir(worker_test_dir)
        self.clear_deck_dir(worker_pool_dir)

        self.write_deck(worker_test_dir, job["deck_name"], job["deck"])
        games = int(job["num_games"])
        for game_index in range(games):
            opponent_copy = worker_pool_dir / f"{job['opponent_path'].stem}_game_{game_index:03d}.dck"
            shutil.copy2(job["opponent_path"], opponent_copy)

        command = [
            str(self.java_exe),
            "-Xmx1536m",
            "-jar",
            str(self.jar_file),
            "sim",
            "-t",
            "DeckTest",
            "-testDir",
            str(worker_test_dir),
            "-poolDir",
            str(worker_pool_dir),
            "-s",
            str(games),
            "-m",
            "1",
            "-q",
            "-c",
            str(timeout),
        ]

        if seed is not None:
            command += ["-seed", str(seed + job["job_id"])]

        result = subprocess.run(
            command,
            cwd=self.working_dir,
            capture_output=True,
            text=True,
            creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Deck test failed with exit code {result.returncode}\n"
                f"STDOUT:\n{result.stdout[-2000:]}\n"
                f"STDERR:\n{result.stderr[-2000:]}"
            )

        results_by_deck = self._parse_deck_results(result.stdout)
        stats = results_by_deck[job["deck_name"]]
        if stats["matches"] != games:
            raise RuntimeError(
                f"Expected {games} raw games for {job['deck_name']} vs {job['opponent_path'].name}, "
                f"Forge reported {stats['matches']}."
            )
        return job["deck_index"], {
            "opponent_index": job["opponent_index"],
            "opponent": job["opponent_path"].name,
            "games": stats["matches"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "ties": stats["ties"],
            "win_rate": stats["win_percentage"] / 100.0,
        }

    def _make_jobs(self, decks, num_games, workers):
        if len(decks) >= workers:
            chunks = [[] for _ in range(workers)]
            for deck_index, deck in enumerate(decks):
                deck_name = f"deck_{deck_index:04d}"
                chunks[deck_index % workers].append((deck_index, deck, deck_name))

            return [
                {"job_id": job_id, "num_games": num_games, "decks": chunk}
                for job_id, chunk in enumerate(chunks)
                if chunk
            ]

        jobs = []
        n_decks = len(decks)
        for deck_index, deck in enumerate(decks):
            shard_count = workers // n_decks
            if deck_index < workers % n_decks:
                shard_count += 1
            shard_count = max(1, min(shard_count, num_games))

            base_games = num_games // shard_count
            extra_games = num_games % shard_count
            for shard_index in range(shard_count):
                shard_games = base_games + (1 if shard_index < extra_games else 0)
                deck_name = f"deck_{deck_index:04d}_shard_{shard_index:02d}"
                jobs.append({
                    "job_id": len(jobs),
                    "num_games": shard_games,
                    "decks": [(deck_index, deck, deck_name)],
                })

        return jobs

    def _parse_deck_results(self, output):
        lines = output.splitlines()

        for i, line in enumerate(lines):
            if line.strip() == "Deck Test Results":
                result_lines = lines[i + 2:]
                break
        else:
            raise RuntimeError("Could not find Deck Test Results in Forge output")

        deck_results = {}
        for line in result_lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            deck_name = " ".join(parts[:-5])
            try:
                deck_results[deck_name] = {
                    "matches": int(parts[-5]),
                    "wins": int(parts[-4]),
                    "losses": int(parts[-3]),
                    "ties": int(parts[-2]),
                    "win_percentage": float(parts[-1]),
                }
            except ValueError as exc:
                raise RuntimeError(f"Could not parse Forge deck result line: {line!r}") from exc

        return deck_results

    def add_deck_to_test_dir(self, deck_name, cards):
        return self.write_deck(self.test_dir, deck_name, cards)

    def write_deck(self, deck_dir, deck_name, cards):
        deck_dir = Path(deck_dir)
        deck_dir.mkdir(parents=True, exist_ok=True)

        file_name = deck_name.replace(" ", "_").lower() + ".dck"
        deck_path = deck_dir / file_name

        lines = [
            "[metadata]",
            f"Name={deck_name}",
            "[Main]",
        ]

        if cards and isinstance(cards[0], str) and cards[0].split()[0].isdigit():
            lines.extend(cards)
        else:
            counted_cards = Counter(cards)
            for card, count in sorted(counted_cards.items()):
                lines.append(f"{count} {card}")

        deck_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(deck_path)

    def clear_test_dir(self):
        self.clear_deck_dir(self.test_dir)

    def clear_deck_dir(self, deck_dir):
        deck_dir = Path(deck_dir)
        deck_dir.mkdir(parents=True, exist_ok=True)

        for deck_file in deck_dir.glob("*.dck"):
            deck_file.unlink()

if __name__ == "__main__":
    tester = deck_tester()
    deck = [
    "2 Additive Evolution",
    "1 Adventurous Eater",
    "1 Berta, Wise Extrapolator",
    "2 Essenceknit Scholar",
    "6 Forest",
    "3 Grapple with Death",
    "1 Infirmary Healer",
    "1 Island",
    "1 Last Gasp",
    "2 Mindful Biomancer",
    "2 Paradox Gardens",
    "2 Pest Mascot",
    "1 Professor Dellian Fel",
    "2 Studious First-Year",
    "7 Swamp",
    "1 Teacher's Pest",
    "1 Thornfist Striker",
    "1 Titan's Grave",
    "1 Tragedy Feaster",
    "1 Wander Off",
    "1 Witherbloom Charm"
    ]

    times = []
    for i in range(1):
        t = time.perf_counter()
        print(tester.test_batch([deck], num_games=25, best_of=1, workers=8,timeout=10))
        et = time.perf_counter() - t
        times.append(et)
    print(times)
