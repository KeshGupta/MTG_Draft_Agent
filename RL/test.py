import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from RL.deck_tester.evaluate_agent import evaluate_agent
from SL.NN import DraftScorer

checkpoint = torch.load(ROOT / "SL" / "models" / "draft_scorer_best.pt", map_location="cpu")
agent = DraftScorer(**checkpoint["model_kwargs"])
agent.load_state_dict(checkpoint["model_state"])
agent.eval()

result = evaluate_agent(
    agent,
    num_drafts=1,
    games_per_deck=25,
    seed=0,
    card_names=checkpoint["card_names"],
    target_deck_size=checkpoint["model_kwargs"].get("target_deck_size", 40),
    fixed_opponents=True,
    opponents_per_deck=5,
    games_per_opponent=5,
    return_details=True,
)

print(f"aggregate_win_rate: {result['win_rate']:.3f}")
print(f"pool_dir: {result['pool_dir']}")
for deck_index, (win_rate, summary, deck) in enumerate(
    zip(result["deck_win_rates"], result["deck_summaries"], result["decks"])
):
    stats = result["tester"]["per_deck"][deck_index]
    print(
        f"deck_{deck_index:04d}: win_rate={win_rate:.3f} "
        f"record={stats['wins']}-{stats['losses']}-{stats['ties']} "
        f"games={stats['games']} lands={summary['lands']} nonlands={summary['nonlands']} "
        f"basics={summary['basic_lands']} colors={summary['nonland_colors']}"
    )
    print("opponents:", ", ".join(result["tester"]["opponents"]))
    print("decklist:")
    for line in deck:
        print(f"  {line}")
