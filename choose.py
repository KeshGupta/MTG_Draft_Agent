import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from NN import DraftScorer

# ---- config (edit these instead of argparse) ----
DATA_DIR = Path(__file__).resolve().parent / "data" / "cleaned"
CARD_JSON = Path(__file__).resolve().parent / "data" / "raw" / "SOS_cards.json"
OUT = Path(__file__).resolve().parent / "models" / "draft_scorer.pt"

USE_CARD_FEATURES = True
LIMIT_CHUNKS = None      # set to e.g. 3 for a quick smoke test
EPOCHS = 8
BATCH = 2048
LR = 2e-3
VAL_FRACTION = 0.08
SEED = 42
# ------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

meta = json.loads((DATA_DIR / "metadata.json").read_text())
card_names = meta["card_names"]
C = len(card_names)

chunks = sorted(DATA_DIR.glob("draft_xy_chunk_*.npz"))[:LIMIT_CHUNKS]


def build_card_features():
    """Color / type / rarity / cmc features per card. Returns (C, F) tensor."""
    if not USE_CARD_FEATURES or not CARD_JSON.exists():
        return None

    lookup = {}
    for card in json.loads(CARD_JSON.read_text()):
        if name := card.get("name"):
            lookup.setdefault(name, card)
            for part in name.split(" // "):
                lookup.setdefault(part, card)

    colors = ["W", "U", "B", "R", "G"]
    types = ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Land", "Planeswalker"]
    rarities = ["common", "uncommon", "rare", "mythic"]

    rows, missing = [], 0
    for name in card_names:
        card = lookup.get(name)
        if card is None:
            card, missing = {}, missing + 1
        cc = set(card.get("colors") or [])
        tl = card.get("type_line") or ""

        def num(key):
            try:
                return min(float(card.get(key) or 0), 12.0) / 12.0
            except (TypeError, ValueError):
                return 0.0

        rows.append(
            [1.0 if c in cc else 0.0 for c in colors]
            + [1.0 if len(cc) > 1 else 0.0]
            + [1.0 if t in tl else 0.0 for t in types]
            + [1.0 if card.get("rarity") == r else 0.0 for r in rarities]
            + [num("cmc"), num("power"), num("toughness")]
            + [1.0 if not card else 0.0]
        )

    print(f"card features: {C - missing}/{C} matched")
    return torch.tensor(rows, dtype=torch.float32)


def val_mask(n, chunk_idx):
    """Deterministic per-row train/val assignment, stable across epochs."""
    rng = np.random.default_rng(SEED + chunk_idx * 1_000_003)
    return rng.random(n) < VAL_FRACTION


def batches(split, epoch):
    order = list(enumerate(chunks))
    rng = np.random.default_rng(SEED + epoch)
    if split == "train":
        rng.shuffle(order)

    for idx, path in order:
        with np.load(path) as d:
            X, y = d["X"], d["y"]
            m = val_mask(len(y), idx)
            rows = np.flatnonzero(~m if split == "train" else m)
            if split == "train":
                rng.shuffle(rows)

            for i in range(0, len(rows), BATCH):
                sel = rows[i : i + BATCH]
                x = torch.as_tensor(X[sel], dtype=torch.float32, device=device)
                yield (
                    x[:, 2 + C : 2 + 2 * C],          # pool counts
                    x[:, 2 : 2 + C] > 0,              # pack mask
                    x[:, 0],                          # pack number
                    x[:, 1],                          # pick number
                    torch.as_tensor(y[sel], dtype=torch.long, device=device),
                )


def run(split, epoch, model, opt=None):
    model.train() if opt else model.eval()
    tot = n = c1 = c3 = nt_c = nt_n = 0

    for pool, mask, pno, kno, tgt in batches(split, epoch):
        with torch.set_grad_enabled(opt is not None):
            logits = model(pool, mask, pno, kno)
            loss = F.cross_entropy(logits, tgt, label_smoothing=0.02 if opt else 0.0)

        if opt:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        b = tgt.numel()
        tot += loss.item() * b
        n += b
        pred = logits.argmax(1)
        c1 += (pred == tgt).sum().item()
        c3 += (logits.topk(3, 1).indices == tgt[:, None]).any(1).sum().item()

        hard = mask.sum(1) > 1                        # picks with a real choice
        nt_n += hard.sum().item()
        nt_c += (pred[hard] == tgt[hard]).sum().item()

    return {
        "loss": tot / max(n, 1),
        "top1": c1 / max(n, 1),
        "top3": c3 / max(n, 1),
        "nontrivial": nt_c / max(nt_n, 1),
    }


model = DraftScorer(C, build_card_features()).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

print(f"device {device} | cards {C} | chunks {len(chunks)}")
best = float("inf")

for epoch in range(EPOCHS):
    tr = run("train", epoch, model, opt)
    va = run("val", epoch, model)
    print(
        f"epoch {epoch+1}/{EPOCHS} | "
        f"train loss {tr['loss']:.4f} top1 {tr['top1']:.4f} | "
        f"val loss {va['loss']:.4f} top1 {va['top1']:.4f} "
        f"top3 {va['top3']:.4f} nontriv {va['nontrivial']:.4f}"
    )

    if va["loss"] < best:
        best = va["loss"]
        OUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "card_names": card_names}, OUT)
        print(f"  saved (val loss {best:.4f})")
