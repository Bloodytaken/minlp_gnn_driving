"""Train the variable–factor GNN on canonical SCIP labels."""

from __future__ import annotations

import argparse
import copy
import os
import pickle

import numpy as np
import torch

from .factor_decisions import structured_loss, structured_metrics
from .factor_gnn import (FactorBipartiteGNN, graph_data, normalize_graph)


def _stats(graphs, attr: str):
    x = torch.cat([getattr(g, attr) for g in graphs], dim=0)
    return x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)


def load_dataset(paths):
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    samples = []
    for path in paths:
        with open(path, "rb") as handle:
            samples.extend(pickle.load(handle)["samples"])
    if not samples:
        raise ValueError("empty dataset")
    if any(s.get("feature_schema") != "factor_v1" for s in samples):
        raise ValueError("all samples must use feature_schema='factor_v1'")
    dims = {(s["variable_features"].shape[1],
             s["constraint_features"].shape[1],
             s["edge_features"].shape[1]) for s in samples}
    if len(dims) != 1:
        raise ValueError(f"mixed feature widths: {sorted(dims)}")
    return samples, [graph_data(s) for s in samples], dims.pop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma-threshold", type=float, default=0.95)
    ap.add_argument("--lane-threshold", type=float, default=0.95)
    ap.add_argument("--root-lane-threshold", type=float, default=1.01)
    ap.add_argument("--split-key", choices=["seed", "sequential"], default="seed")
    ap.add_argument("--val-seed", type=int, default=None)
    ap.add_argument("--val-seeds", type=int, nargs="*", default=None)
    ap.add_argument("--root-lane-weight", type=float, default=2.0)
    ap.add_argument("--no-source-balance", action="store_true")
    args = ap.parse_args()

    if args.epochs < 1:
        ap.error("--epochs must be positive")
    torch.manual_seed(args.seed)
    samples, graphs, (d_v, d_f, d_e) = load_dataset(args.data)
    if args.split_key == "seed" and len({s.get("seed") for s in samples}) > 1:
        seeds = sorted({s.get("seed") for s in samples})
        val_seeds = set(args.val_seeds or
                        ([args.val_seed] if args.val_seed is not None else [seeds[-1]]))
        if not val_seeds.issubset(seeds):
            raise ValueError(f"validation seeds {sorted(val_seeds)} absent from {seeds}")
        train_idx = [i for i, s in enumerate(samples) if s.get("seed") not in val_seeds]
        val_idx = [i for i, s in enumerate(samples) if s.get("seed") in val_seeds]
        print(f"seed-disjoint split: train seeds={sorted(set(seeds) - val_seeds)} "
              f"validation seeds={sorted(val_seeds)}")
    else:
        n_train = max(1, int(0.8 * len(graphs)))
        if len(graphs) > 1:
            n_train = min(n_train, len(graphs) - 1)
        train_idx = list(range(n_train))
        val_idx = list(range(n_train, len(graphs))) or train_idx

    if not train_idx or not val_idx:
        raise ValueError("training and validation splits must both be nonempty")

    v_mean, v_std = _stats([graphs[i] for i in train_idx], "var_feats")
    f_mean, f_std = _stats([graphs[i] for i in train_idx], "factor_feats")
    e_mean, e_std = _stats([graphs[i] for i in train_idx], "edge_attr")
    norm = {"v_mean": v_mean.numpy(), "v_std": v_std.numpy(),
            "f_mean": f_mean.numpy(), "f_std": f_std.numpy(),
            "e_mean": e_mean.numpy(), "e_std": e_std.numpy()}
    for graph in graphs:
        normalize_graph(graph, norm)

    model = FactorBipartiteGNN(d_v, d_f, d_e, hidden=args.hidden,
                               rounds=args.rounds)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    source_count = {}
    for i in train_idx:
        source = samples[i].get("source", "unknown")
        source_count[source] = source_count.get(source, 0) + 1
    source_weight = {
        source: (1.0 if args.no_source_balance else
                 len(train_idx) / (len(source_count) * count))
        for source, count in source_count.items()
    }
    print(f"train source counts={source_count} weights={source_weight}")

    def loss_for(i):
        graph, sample = graphs[i], samples[i]
        loss = structured_loss(model(graph), graph.y.round(),
                               sample["var_name_order"], graph.candidate_mask,
                               root=0, root_lane_weight=args.root_lane_weight)
        return source_weight[sample.get("source", "unknown")] * loss

    def validation_metrics():
        model.eval()
        metrics = {key: dict(groups=0, fixed=0, wrong=0)
                   for key in ("gamma", "lane_future", "lane_root", "all")}
        with torch.no_grad():
            for i in val_idx:
                graph = graphs[i]
                row = structured_metrics(
                    model(graph).cpu().numpy(), graph.y.cpu().numpy().round(),
                    samples[i]["var_name_order"], graph.candidate_mask.cpu().numpy(),
                    root=0, gamma_threshold=args.gamma_threshold,
                    lane_threshold=args.lane_threshold,
                    root_lane_threshold=args.root_lane_threshold)
                for family in metrics:
                    for key in metrics[family]:
                        metrics[family][key] += row[family][key]
        return metrics

    best_key = None
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for i in train_idx:
            opt.zero_grad()
            loss = loss_for(i)
            loss.backward()
            opt.step()
            total += float(loss)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            metrics = validation_metrics()
            gamma = metrics["gamma"]
            error_rate = gamma["wrong"] / max(gamma["fixed"], 1)
            # Treat <=0.5% group disagreement as the calibrated safe band,
            # then maximise useful coverage inside it.
            key = (error_rate > .005, error_rate, -gamma["fixed"])
            if best_key is None or key < best_key:
                best_key, best_epoch = key, epoch
                best_state = copy.deepcopy(model.state_dict())
            print(f"epoch {epoch:03d}/{args.epochs} "
                  f"train_loss={total / len(train_idx):.5f} "
                  f"val_gamma={gamma['fixed']}/{gamma['groups']} "
                  f"wrong={gamma['wrong']}", flush=True)

    model.load_state_dict(best_state)
    metrics = validation_metrics()
    print(f"selected epoch {best_epoch} with key={best_key}")
    for family, row in metrics.items():
        print(f"validation {family:11s} groups={row['groups']:4d} "
              f"fixed={row['fixed']:4d} wrong={row['wrong']:3d} "
              f"coverage={row['fixed']/max(row['groups'],1):.3f} "
              f"wrong/fixed={row['wrong']/max(row['fixed'],1):.3f}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "factor_gnn.pt")
    torch.save({
        "feature_schema": "factor_v1", "model_state": model.state_dict(),
        "d_v": d_v, "d_f": d_f, "d_e": d_e, "hidden": args.hidden,
        "rounds": args.rounds, "normalization": norm,
        "data": [os.path.abspath(path) for path in args.data], "epochs": args.epochs,
        "threshold": args.gamma_threshold,
        "gamma_threshold": args.gamma_threshold,
        "lane_threshold": args.lane_threshold,
        "root_lane_threshold": args.root_lane_threshold,
        "root_lane_weight": args.root_lane_weight,
        "split_key": args.split_key,
        "validation_seeds": sorted(args.val_seeds or
                                   ([args.val_seed] if args.val_seed is not None else [])),
        "selected_epoch": best_epoch,
        "source_balance": not args.no_source_balance,
        "loss_kind": "structured_groups_v1",
    }, path)
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
