"""Multi-round GNN for variable–factor graphs."""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, d_out))


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = src.new_zeros((size, src.shape[-1]))
    count = src.new_zeros((size, 1))
    out.index_add_(0, index, src)
    count.index_add_(0, index, src.new_ones((src.shape[0], 1)))
    return out / count.clamp_min_(1.0)


class FactorGraphData:
    __slots__ = ("var_feats", "factor_feats", "edge_index", "edge_attr",
                 "candidate_mask", "y", "n_vars", "n_factors")


def graph_data(features: dict) -> FactorGraphData:
    data = FactorGraphData()
    data.var_feats = torch.as_tensor(features["variable_features"], dtype=torch.float32)
    data.factor_feats = torch.as_tensor(features["constraint_features"], dtype=torch.float32)
    data.edge_index = torch.as_tensor(features["edge_index"], dtype=torch.long)
    data.edge_attr = torch.as_tensor(features["edge_features"], dtype=torch.float32)
    data.n_vars = int(data.var_feats.shape[0])
    data.n_factors = int(data.factor_feats.shape[0])
    data.candidate_mask = (torch.as_tensor(features["candidate_mask"], dtype=torch.bool)
                           if "candidate_mask" in features else None)
    data.y = (torch.as_tensor(features["solution_values"], dtype=torch.float32)
              if "solution_values" in features else None)
    return data


def normalize_graph(data: FactorGraphData, stats: dict) -> FactorGraphData:
    """Apply checkpointed train-set normalization in place."""
    for attr, mean_key, std_key in (
            ("var_feats", "v_mean", "v_std"),
            ("factor_feats", "f_mean", "f_std"),
            ("edge_attr", "e_mean", "e_std")):
        value = getattr(data, attr)
        mean = torch.as_tensor(stats[mean_key], dtype=value.dtype, device=value.device)
        std = torch.as_tensor(stats[std_key], dtype=value.dtype, device=value.device)
        setattr(data, attr, (value - mean) / std.clamp_min(1e-6))
    return data


class FactorBipartiteGNN(nn.Module):
    """Residual V→F→V message passing with typed multi-dimensional edges."""

    def __init__(self, d_v: int, d_f: int, d_e: int, hidden: int = 64,
                 rounds: int = 3):
        super().__init__()
        if rounds < 2:
            raise ValueError("The factor GNN requires at least two V-F-V rounds")
        self.rounds = int(rounds)
        self.var_embed = _mlp(d_v, hidden, hidden)
        self.factor_embed = _mlp(d_f, hidden, hidden)
        self.msg_vf = _mlp(hidden + d_e, hidden, hidden)
        self.msg_fv = _mlp(hidden + d_e, hidden, hidden)
        self.factor_update = _mlp(2 * hidden, hidden, hidden)
        self.var_update = _mlp(2 * hidden, hidden, hidden)
        self.factor_norm = nn.LayerNorm(hidden)
        self.var_norm = nn.LayerNorm(hidden)
        self.out = _mlp(hidden, hidden, 1)

    def encode(self, data: FactorGraphData):
        """Return final variable and factor embeddings.

        Keeping graph encoding separate lets downstream heads reuse the exact
        frozen representation learned for binary prediction.
        """
        h_v = self.var_embed(data.var_feats)
        h_f = self.factor_embed(data.factor_feats)
        f_idx, v_idx = data.edge_index[0], data.edge_index[1]
        edge = data.edge_attr
        for _ in range(self.rounds):
            msg_f = self.msg_vf(torch.cat([h_v[v_idx], edge], dim=-1))
            agg_f = _scatter_mean(msg_f, f_idx, data.n_factors)
            h_f = self.factor_norm(h_f + self.factor_update(torch.cat([h_f, agg_f], -1)))

            msg_v = self.msg_fv(torch.cat([h_f[f_idx], edge], dim=-1))
            agg_v = _scatter_mean(msg_v, v_idx, data.n_vars)
            h_v = self.var_norm(h_v + self.var_update(torch.cat([h_v, agg_v], -1)))
        return h_v, h_f

    def forward(self, data: FactorGraphData) -> torch.Tensor:
        h_v, _ = self.encode(data)
        return self.out(h_v).squeeze(-1)
