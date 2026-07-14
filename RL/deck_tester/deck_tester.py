import subprocess
from collections import Counter
from pathlib import Path


BASE_DIR = r"C:\Users\samth\OneDrive\Desktop\mtg\MTG_Draft_Agent\RL\deck_tester"
TEST_DIR = BASE_DIR + r"\test"
POOL_DIR = BASE_DIR + r"\pool"
JAVA_EXE = r"C:\Users\samth\.jdks\ms-17.0.19\bin\java.exe"
JAR_FILE = BASE_DIR + r"\forge-gui-desktop-2.0.14-SNAPSHOT-jar-with-dependencies.jar"
WORKING_DIR = r"C:\Users\samth\Downloads\forge\forge-gui"


class deck_tester:
    def __init__(self):
        self.base_dir = BASE_DIR
        self.test_dir = TEST_DIR
        self.pool_dir = POOL_DIR
        self.java_exe = JAVA_EXE
        self.jar_file = JAR_FILE
        self.working_dir = WORKING_DIR

    def test_batch(self, decks, sample_size=5, match_size=1, timeout=300, seed=None):
        self.clear_test_dir()

        deck_names = []
        for i, deck in enumerate(decks):
            deck_name = f"deck_{i:04d}"
            deck_names.append(deck_name)
            self.add_deck_to_test_dir(deck_name, deck)

        command = [
            self.java_exe,
            "-Xmx4096m",
            "-jar",
            self.jar_file,
            "sim",
            "-t",
            "DeckTest",
            "-testDir",
            self.test_dir,
            "-poolDir",
            self.pool_dir,
            "-s",
            str(sample_size),
            "-m",
            str(match_size),
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
        )

        print(result.stdout)
        print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(f"Deck test failed with exit code {result.returncode}")

        results_by_deck = self._parse_deck_results(result.stdout)
        return [results_by_deck[deck_name] for deck_name in deck_names]

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
        test_dir = Path(self.test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)

        file_name = deck_name.replace(" ", "_").lower() + ".dck"
        deck_path = test_dir / file_name

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
        print(f"Created deck: {deck_path}")
        return str(deck_path)

    def clear_test_dir(self):
        test_dir = Path(self.test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)

        for deck_file in test_dir.glob("*.dck"):
            deck_file.unlink()

        print(f"Cleared .dck files from: {self.test_dir}")



tester = deck_tester()
deck = [
    "Burnout Bashtronaut",
    "Burnout Bashtronaut",
    "Burnout Bashtronaut",
    "Burnout Bashtronaut",

    "Burst Lightning",
    "Burst Lightning",
    "Burst Lightning",
    "Burst Lightning",

    "Hexing Squelcher",
    "Hexing Squelcher",
    "Hexing Squelcher",
    "Hexing Squelcher",

    "Hired Claw",
    "Hired Claw",
    "Hired Claw",
    "Hired Claw",

    "Howlsquad Heavy",
    "Howlsquad Heavy",
    "Howlsquad Heavy",
    "Howlsquad Heavy",

    "Lightning Strike",
    "Lightning Strike",
    "Lightning Strike",

    "Magebane Lizard",
    "Magebane Lizard",
    "Magebane Lizard",
    "Magebane Lizard",

    "Nova Hellkite",
    "Nova Hellkite",
    "Nova Hellkite",

    "Shock",
    "Shock",
    "Shock",
    "Shock",

    "Sunspine Lynx",
    "Sunspine Lynx",

    "Rockface Village",
    "Rockface Village",

    "Soulstone Sanctuary",
    "Soulstone Sanctuary",
    "Soulstone Sanctuary",

    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
    "Mountain",
]
print(tester.test_batch([deck], sample_size=5, match_size=1, timeout=300))
