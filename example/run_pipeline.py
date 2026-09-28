"""Generate SCIP labels, train the GNN, and solve a held-out reduced MINLP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import torch

from minlp_builder import MINLPBuilder
from minlp_tree import build_dual_tree
from offline_gnn.factor_decisions import structured_fixmap
from offline_gnn.factor_gnn import FactorBipartiteGNN, graph_data, normalize_graph
from offline_gnn.features_factor import extract_factor_features
from offline_gnn.root_lp import configure_feature_pass, features_sane
from problem_setup import make_weights, straight_road_geom

ROOT = Path(__file__).resolve().parents[1]
H, HB, DT = 2, 1, 0.2


def make_instance(seed):
    rng = np.random.default_rng(seed)
    return {
        "seed": seed,
        "ego": [0.0, float(rng.uniform(7.7, 8.3)), 1.75, 0.0],
        "opponent": [float(rng.uniform(10.0, 11.0)),
                     float(rng.uniform(6.7, 7.3)), 1.75, 0.0],
        "prior_agg": float(rng.uniform(0.3, 0.7)),
    }


def build(instance):
    geom = straight_road_geom(H=H, M=3, lane0=0)
    weights = make_weights(H, [0.0, 1.75], geom, v_nom=8.0)
    prior = {"agg": instance["prior_agg"], "cau": 1.0-instance["prior_agg"]}
    tree, opponent, _ = build_dual_tree(
        H, HB, DT, instance["ego"], instance["opponent"], prior=prior)
    builder = MINLPBuilder(H, DT, weights, geom)
    return builder, builder.build(instance["ego"], tree, None, opponent)


def extract_graph(instance, time_limit):
    builder, model = build(instance)
    handler = configure_feature_pass(model, time_limit=time_limit)
    model.optimize()
    if handler.fired == 0:
        raise RuntimeError(f"Root LP was not solved: {model.getStatus()}")
    features = extract_factor_features(model, builder)
    if not features_sane(features) or features["missing_nonlinear_factors"]:
        raise RuntimeError("Invalid or incomplete MINLP graph")
    return features


def solve(builder, model, time_limit):
    model.setParam("limits/time", time_limit)
    started = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - started
    status = str(model.getStatus())
    if model.getNSols() == 0:
        raise RuntimeError(f"No feasible solution: {status}")
    sol = model.getBestSol()
    return {
        "status": status,
        "objective": float(model.getSolObjVal(sol)),
        "gap": float(model.getGap()),
        "solve_seconds": elapsed,
        "root_control": [float(model.getSolVal(sol, v)) for v in builder.U[0]],
    }


def collect_sample(instance, time_limit):
    features = extract_graph(instance, time_limit)
    builder, model = build(instance)
    result = solve(builder, model, time_limit)
    if result["status"] != "optimal":
        raise RuntimeError(f"Seed {instance['seed']}: optimal labels required; {result}")
    sol = model.getBestSol()
    values = {v.name: float(model.getSolVal(sol, v)) for v in model.getVars()}
    features["solution_values"] = np.asarray(
        [values[name] for name in features["var_name_order"]], dtype=np.float32)
    features.update(feature_schema="factor_v1", seed=instance["seed"],
                    source="example", solve=result)
    return features


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--time-limit", type=float, default=15.0)
    ap.add_argument("--output", type=Path, default=ROOT / "example" / "output")
    ap.add_argument("--gamma-threshold", type=float, default=None,
                    help="Inference region confidence; default: checkpoint value (0.95)")
    ap.add_argument("--lane-threshold", type=float, default=None,
                    help="Inference future-lane confidence; default: checkpoint value (0.95)")
    ap.add_argument("--root-lane-threshold", type=float, default=None,
                    help="Inference root-lane confidence; default: checkpoint value (1.01, disabled)")
    args = ap.parse_args()
    for key in ("gamma_threshold", "lane_threshold", "root_lane_threshold"):
        value = getattr(args, key)
        if value is not None and (not np.isfinite(value) or value < 0):
            ap.error(f"{key} must be finite and nonnegative; values > 1 disable fixing")
    if args.epochs < 1 or args.time_limit <= 0:
        ap.error("epochs and time-limit must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    train_seeds, val_seeds, test_seed = list(range(10)), [1000, 1001], 2000

    print("[1/4] Collecting 10 training and 2 validation MINLPs", flush=True)
    samples = []
    for seed in train_seeds + val_seeds:
        sample = collect_sample(make_instance(seed), args.time_limit)
        samples.append(sample)
        print(f"  seed={seed}: {sample['solve']['status']}, "
              f"variables={len(sample['var_name_order'])}", flush=True)
    dataset = output / "dataset.pkl"
    with dataset.open("wb") as handle:
        pickle.dump({"samples": samples}, handle)

    print("[2/4] Training with the existing offline_gnn entry point", flush=True)
    command = [sys.executable, "-m", "offline_gnn.train_factor_gnn",
               "--data", str(dataset), "--save-dir", str(output),
               "--epochs", str(args.epochs), "--hidden", "32", "--rounds", "2",
               "--val-seeds", *map(str, val_seeds)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    with (output / "training.log").open("w") as log:
        trained = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
    if trained.returncode:
        raise RuntimeError((output / "training.log").read_text())
    print((output / "training.log").read_text(), end="", flush=True)

    print("[3/4] Predicting decisions for unseen seed 2000", flush=True)
    checkpoint = torch.load(output / "factor_gnn.pt", map_location="cpu", weights_only=False)
    network = FactorBipartiteGNN(
        checkpoint["d_v"], checkpoint["d_f"], checkpoint["d_e"],
        hidden=checkpoint["hidden"], rounds=checkpoint["rounds"])
    network.load_state_dict(checkpoint["model_state"])
    network.eval()
    thresholds = {key: (getattr(args, key) if getattr(args, key) is not None
                        else checkpoint[key])
                  for key in ("gamma_threshold", "lane_threshold", "root_lane_threshold")}
    instance = make_instance(test_seed)
    started = time.perf_counter()
    features = extract_graph(instance, args.time_limit)
    feature_seconds = time.perf_counter() - started
    graph = normalize_graph(graph_data(features), checkpoint["normalization"])
    started = time.perf_counter()
    with torch.no_grad():
        logits = network(graph).numpy()
    fixmap, decisions, n_region, n_lane = structured_fixmap(
        logits, features["var_name_order"], features["candidate_mask"], root=0,
        **thresholds)
    inference_seconds = time.perf_counter() - started
    if not fixmap:
        print("No decisions pass the thresholds; all candidates remain free.", flush=True)

    print("[4/4] Solving full and GNN-reduced held-out MINLPs", flush=True)
    full_builder, full_model = build(instance)
    full = solve(full_builder, full_model, args.time_limit)
    reduced_builder, reduced_model = build(instance)
    variables = {v.name: v for v in reduced_model.getVars()}
    for name, value in fixmap.items():
        infeasible, fixed = reduced_model.fixVar(variables[name], value)
        if infeasible or not fixed:
            raise RuntimeError(f"Could not fix {name}={value}")
    reduced = solve(reduced_builder, reduced_model, args.time_limit)
    if full["status"] != "optimal" or reduced["status"] != "optimal":
        raise RuntimeError("The small example requires both solves to reach optimality")
    full_sol = full_model.getBestSol()
    baseline_values = {v.name: full_model.getSolVal(full_sol, v) for v in full_model.getVars()}
    report = {
        "purpose": "Pipeline smoke example; not a speedup or generalization benchmark",
        "horizon": H, "branching_horizon": HB, "dt": DT,
        "train_seeds": train_seeds, "validation_seeds": val_seeds,
        "test_instance": instance, "selected_epoch": checkpoint["selected_epoch"],
        "graph": {k: list(features[k].shape) for k in
                  ["variable_features", "constraint_features", "edge_features"]},
        "inference_thresholds": thresholds,
        "decision_fix_rate": len(decisions) / max(n_region+n_lane, 1),
        "candidate_binary_fix_rate": len(fixmap) / max(int(features["candidate_mask"].sum()), 1),
        "fixed_decisions": len(decisions), "total_decisions": n_region+n_lane,
        "fixed_binaries": len(fixmap),
        "candidate_binaries": int(features["candidate_mask"].sum()),
        "fixmap": fixmap, "decisions": decisions,
        "binary_disagreements_with_full_solution": sum(
            int(round(baseline_values[name]) != value) for name, value in fixmap.items()),
        "full": full, "reduced": reduced,
        "objective_difference": reduced["objective"]-full["objective"],
        "feature_seconds_including_build": feature_seconds,
        "inference_and_decoding_seconds": inference_seconds,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Fixed {len(decisions)}/{n_region+n_lane} decisions "
          f"({len(fixmap)} binaries), objective difference="
          f"{report['objective_difference']:.6g}", flush=True)
    print(f"Results: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
