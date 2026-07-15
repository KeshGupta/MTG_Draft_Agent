import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

class deck_tester:
    def __init__(self):
        self.base_dir = r"C:\Users\samth\OneDrive\Desktop\mtg\MTG_Draft_Agent\RL\deck_tester"
        self.test_dir = r"C:\Users\samth\OneDrive\Desktop\mtg\MTG_Draft_Agent\RL\deck_tester\test"
        self.pool_dir = r"C:\Users\samth\OneDrive\Desktop\mtg\MTG_Draft_Agent\RL\deck_tester\pool"
        self.java_exe = r"C:\Users\samth\.jdks\ms-17.0.19\bin\java.exe"
        self.jar_file = r"C:\Users\samth\OneDrive\Desktop\mtg\MTG_Draft_Agent\RL\deck_tester\forge-gui-desktop-2.0.14-SNAPSHOT-jar-with-dependencies.jar"
        self.working_dir = r"C:\Users\samth\Downloads\forge\forge-gui"

    def test_batch(self, decks, num_games=5, best_of=1, timeout=300, seed=None, workers=4):
        if not decks:
            return []

        workers = max(1, min(workers, len(decks)))
        chunks = self._split_decks(decks, workers)
        results = [None] * len(decks)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for worker_id, chunk in enumerate(chunks):
                if not chunk:
                    continue

                futures.append(executor.submit(
                    self._run_deck_chunk,
                    worker_id,
                    chunk,
                    num_games,
                    best_of,
                    timeout,
                    seed,
                ))

            for future in as_completed(futures):
                for deck_index, win_percentage in future.result():
                    results[deck_index] = win_percentage

        return results

    def _run_deck_chunk(self, worker_id, deck_chunk, num_games, best_of, timeout, seed):
        worker_test_dir = Path(self.base_dir) / "worker_tests" / f"worker_{worker_id:02d}"
        self.clear_deck_dir(worker_test_dir)

        deck_names = []
        for deck_index, deck in deck_chunk:
            deck_name = f"deck_{deck_index:04d}"
            deck_names.append(deck_name)
            self.write_deck(worker_test_dir, deck_name, deck)

        command = [
            self.java_exe,
            "-Xmx1536m",
            "-jar",
            self.jar_file,
            "sim",
            "-t",
            "DeckTest",
            "-testDir",
            str(worker_test_dir),
            "-poolDir",
            self.pool_dir,
            "-s",
            str(num_games),
            "-m",
            str(best_of),
            "-q",
            "-c",
            str(timeout),
        ]

        if seed is not None:
            command += ["-seed", str(seed)]

        result = subprocess.run(
            command,
            cwd=self.working_dir,
            capture_output=True,
            text=True,
            creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Deck test failed with exit code {result.returncode}")

        results_by_deck = self._parse_deck_results(result.stdout)
        return [
            (deck_index, results_by_deck[deck_name])
            for (deck_index, _), deck_name in zip(deck_chunk, deck_names)
        ]

    def _split_decks(self, decks, workers):
        chunks = [[] for _ in range(workers)]
        for deck_index, deck in enumerate(decks):
            chunks[deck_index % workers].append((deck_index, deck))
        return chunks

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
            win_percentage = float(parts[-1])
            deck_results[deck_name] = win_percentage

        return deck_results

    def add_deck_to_test_dir(self, deck_name, cards):
        return self.write_deck(Path(self.test_dir), deck_name, cards)

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
        self.clear_deck_dir(Path(self.test_dir))

    def clear_deck_dir(self, deck_dir):
        deck_dir = Path(deck_dir)
        deck_dir.mkdir(parents=True, exist_ok=True)

        for deck_file in deck_dir.glob("*.dck"):
            deck_file.unlink()

if __name__ == "__main__":
    tester = deck_tester()
    deck = [
        "Ajani's Response",
        "Ascendant Dustspeaker",
        "Aziza, Mage Tower Captain",
        "Colossus of the Blood Age",
        "Colossus of the Blood Age",
        "Daydream",
        "Dig Site Inventory",
        "Eager Glyphmage",
        "Elite Interceptor",
        "Fields of Strife",
        "Garrison Excavator",
        "Group Project",
        "Lorehold Charm",
        "Molten Note",
        "Monstrous Rage",
        "Mountain",
        "Mountain",
        "Mountain",
        "Mountain",
        "Mountain",
        "Mountain",
        "Mountain",
        "Mountain",
        "Plains",
        "Plains",
        "Plains",
        "Plains",
        "Plains",
        "Plains",
        "Plains",
        "Plains",
        "Practiced Scrollsmith",
        "Practiced Scrollsmith",
        "Pursue the Past",
        "Rehearsed Debater",
        "Reprieve",
        "Rubble Rouser",
        "Shattered Acolyte",
        "Stone Docent",
        "Stone Docent",
        "Wilt in the Heat",
    ]

    times = []
    for i in range(4):
        t = time.perf_counter()
        print(tester.test_batch([deck, deck, deck,deck], num_games=25, best_of=1, workers=4,timeout=10,seed=5))
        et = time.perf_counter() - t
        times.append(et)
    print(times)
