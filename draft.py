import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from NN import DraftScorer

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "cleaned"

meta = json.load(open(DATA_DIR / "metadata.json"))
card_names = meta["card_names"]
num_cards = len(card_names)

# column layout: [pack_number, pick_number, pack_counts(C), pool_counts(C)]
pack_start = 2
pack_end = 2 + num_cards
pool_start = pack_end
pool_end = pool_start + num_cards

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chunk_paths = sorted(DATA_DIR.glob("draft_xy_chunk_*.npz"))

# chunk-level holdout instead of row shuffling (avoids same-draft leakage)
train_paths = chunk_paths[0:2]
test_paths = chunk_paths[3:4]


def load(paths):
    xs = [np.load(p)["X"] for p in paths]
    ys = [np.load(p)["y"] for p in paths]
    return np.concatenate(xs), np.concatenate(ys)


def to_tensors(x_np, y_np):
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).long()
    pool_counts = x[:, pool_start:pool_end]
    pack_mask = x[:, pack_start:pack_end] > 0
    return pool_counts, pack_mask, y


x_train_np, y_train_np = load(train_paths)
x_test_np, y_test_np = load(test_paths)

pool_tr, mask_tr, y_tr = to_tensors(x_train_np, y_train_np)
pool_te, mask_te, y_te = to_tensors(x_test_np, y_test_np)

train_loader = DataLoader(
    TensorDataset(pool_tr, mask_tr, y_tr),
    batch_size=256, shuffle=True, drop_last=True,
)
test_loader = DataLoader(
    TensorDataset(pool_te, mask_te, y_te),
    batch_size=256,
)

model = DraftScorer(num_cards=num_cards, emb_dim=64).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)


def evaluate(loader):
    model.eval()
    correct = correct3 = samples = 0
    with torch.no_grad():
        for pool, mask, yb in loader:
            pool, mask, yb = pool.to(device), mask.to(device), yb.to(device)
            logits = model(pool, mask)
            correct += (logits.argmax(1) == yb).sum().item()
            top3 = logits.topk(3, dim=1).indices
            correct3 += (top3 == yb.unsqueeze(1)).any(1).sum().item()
            samples += yb.size(0)
    return correct / samples, correct3 / samples


EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss, train_correct, train_samples = 0, 0, 0

    for pool, mask, yb in train_loader:
        pool, mask, yb = pool.to(device), mask.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(pool, mask)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_correct += (logits.argmax(1) == yb).sum().item()
        train_samples += yb.size(0)

    train_acc = train_correct / train_samples
    test_acc, test_top3 = evaluate(test_loader)
    scheduler.step(total_loss / len(train_loader))

    print(
        f"Epoch {epoch + 1}/{EPOCHS} - "
        f"loss: {total_loss / len(train_loader):.4f} - "
        f"train acc: {train_acc:.4f} - "
        f"test acc: {test_acc:.4f} - test top3: {test_top3:.4f}"
    )