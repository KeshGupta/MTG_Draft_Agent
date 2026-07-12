import torch
from torch import nn


class DraftScorer(nn.Module):
    """Fast multi-output baseline using the one-hot/count draft state directly."""

    def __init__(
        self,
        num_cards,
        emb_dim=128,
        hidden_dim=256,
        dropout=0.15,
        max_packs=3,
        max_picks=14,
        **_,
    ):
        super().__init__()
        self.num_cards = num_cards
        self.max_packs = max_packs
        self.max_picks = max_picks

        self.net = nn.Sequential(
            nn.LayerNorm(num_cards * 2 + 4),
            nn.Linear(num_cards * 2 + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_cards),
        )

    def forward(self, pool_counts, pack_counts, pack_numbers=None, pick_numbers=None):
        pool_counts = pool_counts.float()
        pack_counts = pack_counts.float()
        pack_mask = pack_counts > 0

        batch_size = pool_counts.shape[0]
        device = pool_counts.device
        if pack_numbers is None:
            pack_numbers = torch.zeros(batch_size, device=device)
        if pick_numbers is None:
            pick_numbers = torch.zeros(batch_size, device=device)

        stage = torch.stack(
            [
                pack_numbers.float() / max(self.max_packs - 1, 1),
                pick_numbers.float() / max(self.max_picks - 1, 1),
                pool_counts.sum(1) / 42.0,
                pack_counts.sum(1) / 14.0,
            ],
            dim=1,
        )

        logits = self.net(torch.cat([pool_counts, pack_counts, stage], dim=1))
        return logits.masked_fill(~pack_mask, -1e9)


class _ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ContextualDraftScorer(nn.Module):
    """Scores each legal pack card from the drafted pool, pack context, and pick stage."""

    def __init__(
        self,
        num_cards,
        card_features=None,
        emb_dim=128,
        hidden_dim=256,
        synergy_dim=64,
        dropout=0.15,
        max_packs=3,
        max_picks=14,
    ):
        super().__init__()
        if card_features is None:
            card_features = torch.empty(num_cards, 0)
        if card_features.ndim != 2 or card_features.shape[0] != num_cards:
            raise ValueError("card_features must have shape (num_cards, feature_dim).")

        self.num_cards = num_cards
        self.emb_dim = emb_dim
        self.synergy_dim = synergy_dim
        self.max_packs = max_packs
        self.max_picks = max_picks

        self.register_buffer("card_features", card_features.float().clone())
        self.card_emb = nn.Embedding(num_cards, emb_dim)
        self.card_feature_proj = (
            nn.Linear(card_features.shape[1], emb_dim, bias=False)
            if card_features.shape[1] > 0
            else None
        )
        self.card_norm = nn.LayerNorm(emb_dim)
        self.card_key = nn.Linear(emb_dim, emb_dim, bias=False)

        self.pack_number_emb = nn.Embedding(max_packs, emb_dim)
        self.pick_number_emb = nn.Embedding(max_picks, emb_dim)

        context_input_dim = emb_dim * 4 + 4
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(context_input_dim),
            nn.Linear(context_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            _ResidualBlock(hidden_dim, dropout),
            _ResidualBlock(hidden_dim, dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.query = nn.Linear(emb_dim, emb_dim, bias=False)

        self.card_bias = nn.Parameter(torch.zeros(num_cards))
        self.stage_card_bias = nn.Embedding(max_packs * max_picks, num_cards)
        nn.init.zeros_(self.stage_card_bias.weight)

        self.pool_synergy = nn.Parameter(torch.empty(num_cards, synergy_dim))
        self.candidate_synergy = nn.Parameter(torch.empty(num_cards, synergy_dim))
        nn.init.normal_(self.pool_synergy, mean=0.0, std=0.02)
        nn.init.normal_(self.candidate_synergy, mean=0.0, std=0.02)
        self.synergy_scale = nn.Parameter(torch.tensor(0.1))

    def _card_representations(self):
        card_vecs = self.card_emb.weight
        if self.card_feature_proj is not None:
            card_vecs = card_vecs + self.card_feature_proj(self.card_features)
        return self.card_norm(card_vecs)

    def forward(self, pool_counts, pack_counts, pack_numbers=None, pick_numbers=None):
        # pool_counts:  (B, C) counts of already drafted cards
        # pack_counts:  (B, C) counts of cards currently available
        # pack_numbers: (B,) pack index, usually 0..2
        # pick_numbers: (B,) pick index, usually 0..13
        pool_counts = pool_counts.float()
        pack_counts = pack_counts.float()
        pack_mask = pack_counts > 0

        batch_size = pack_counts.shape[0]
        device = pack_counts.device
        if pack_numbers is None:
            pack_numbers = torch.zeros(batch_size, dtype=torch.long, device=device)
        if pick_numbers is None:
            pick_numbers = torch.zeros(batch_size, dtype=torch.long, device=device)
        pack_numbers = pack_numbers.long().clamp(0, self.max_packs - 1)
        pick_numbers = pick_numbers.long().clamp(0, self.max_picks - 1)

        card_vecs = self._card_representations()
        logged_pool = torch.log1p(pool_counts)
        pool_weight = logged_pool.sum(dim=1, keepdim=True).clamp_min(1.0)
        pack_weight = pack_counts.sum(dim=1, keepdim=True).clamp_min(1.0)

        pool_sum = (logged_pool @ card_vecs) / pool_weight.sqrt()
        pool_avg = (logged_pool @ card_vecs) / pool_weight
        pack_avg = (pack_counts @ card_vecs) / pack_weight
        stage_vec = self.pack_number_emb(pack_numbers) + self.pick_number_emb(pick_numbers)

        scalars = torch.cat(
            [
                pack_numbers.float().unsqueeze(1) / max(self.max_packs - 1, 1),
                pick_numbers.float().unsqueeze(1) / max(self.max_picks - 1, 1),
                pool_counts.sum(dim=1, keepdim=True) / 42.0,
                pack_weight / 14.0,
            ],
            dim=1,
        )
        context = torch.cat([pool_sum, pool_avg, pack_avg, stage_vec, scalars], dim=1)
        query = self.query(self.context_mlp(context))
        keys = self.card_key(card_vecs)
        logits = (query @ keys.T) * (self.emb_dim ** -0.5)

        stage_index = pack_numbers * self.max_picks + pick_numbers
        logits = logits + self.card_bias + self.stage_card_bias(stage_index)

        synergy_pool = logged_pool @ self.pool_synergy
        synergy_logits = (synergy_pool @ self.candidate_synergy.T) * (self.synergy_dim ** -0.5)
        logits = logits + self.synergy_scale * synergy_logits

        return logits.masked_fill(~pack_mask, -1e9)
