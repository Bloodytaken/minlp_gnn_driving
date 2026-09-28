# MINLP dual MPC + GNN


## Code map

| Component | Entry points |
| --- | --- |
| MINLP formulation | `minlp_builder.py`: `MINLPBuilder` |
| Ego dynamics, lane decisions, safe regions | `minlp_builder.py`, `road_boundary.py` |
| Scenario tree and endogenous belief update | `minlp_tree.py`, `tree.py`, `belief_law.py` |
| Opponent reaction and following constraints | `opponent_model/opponent_law.py`, `opponent_model/follow_law.py`, `stop_law.py` |
| Numeric model and setup | `opponent_model/opponent.py`, `parameters.py`, `opponent_model/traffic_model.py`, `opponent_model/traffic_follow.py`, `problem_setup.py` |
| Nonlinear relation metadata | `offline_gnn/factor_schema.py` |
| Root-relaxation graph extraction | `offline_gnn/root_lp.py`, `offline_gnn/features_factor.py`, `offline_gnn/feature_utils.py` |
| GNN architecture | `offline_gnn/factor_gnn.py`: `FactorBipartiteGNN` |
| Decision distributions, loss, confidence fixing | `offline_gnn/factor_decisions.py` |
| Training | `offline_gnn/train_factor_gnn.py` |

`minlp_builder.py` assembles the complete formulation in one place: base
motion/lane constraints, background safety regions, then endogenous opponent
reactions and beliefs. `MINLPBuilder` is the public modeling entry point;
its private base classes organize these assembly stages.
Simulation drivers, benchmarks, plots, datasets, checkpoints and historical
snapshots are omitted.

The `opponent_model/` package contains the numeric opponent models and their
SCIP constraint encodings; `offline_gnn/` contains graph features and training.

## Setup and modeling

Tested with Python 3.10, NumPy 1.26.4, PyTorch 2.4.1 and PySCIPOpt 5.7.1.
Run from this directory:

```bash
python -m pip install -r requirements.txt
```

```python
from problem_setup import straight_road_geom, make_weights
from minlp_tree import build_dual_tree
from minlp_builder import MINLPBuilder

H, Hb, dt = 2, 1, 0.2
# Builder inputs: [x, vx, y, vy]; internal tree states: [x, y, vx, vy].
x_ego = [0.0, 8.0, 1.75, 0.0]
x_opp = [12.0, 7.0, 1.75, 0.0]
geom = straight_road_geom(H=H, M=3, lane0=0)
weights = make_weights(H, [0.0, 1.75], geom, v_nom=8.0)
tree, opponent, _ = build_dual_tree(H, Hb, dt, x_ego, x_opp)
builder = MINLPBuilder(H, dt, weights, geom)
model = builder.build(x_ego, tree, None, opponent)
model.setParam("limits/time", 15.0)
model.optimize()
print(model.getStatus())
```

## End-to-end example

Run `python -m example.run_pipeline` from this directory to generate a small
dataset, train the GNN and solve a held-out problem with predicted integer
fixing. See [example/README.md](example/README.md) for the verified results.

## GNN training

The `offline_gnn/` package groups graph extraction, network and training code.
`offline_gnn/factor_schema.py` records nonlinear relations during model
construction so graph extraction can retain their structure. The formulation
imports this lightweight helper; it does not require PyTorch. Run training from `MINLP/` as a Python module:

```bash
python -m offline_gnn.train_factor_gnn \
  --data /path/to/minlp_dataset.pkl \
  --save-dir /path/to/checkpoints --epochs 30
```

The input pickle is a dictionary with a `samples` list. Each sample contains
`feature_schema="factor_v1"`, `variable_features` (V × dv),
`constraint_features` (C × dc), `edge_features` (E × de),
`edge_index` (2 × E, constraint index first), `candidate_mask` (V),
`var_name_order` (V), and `solution_values` (V, SCIP labels in the same order).
Optional `seed` identifies simulation episodes for a seed-disjoint validation
split; `source` identifies data sources for balancing. Without multiple seeds,
the script uses a sequential 80/20 split. Supply multiple independent episodes
for meaningful validation. Supply your own datasets or generate the small example dataset above.

Feature extraction imports are:

```python
from offline_gnn.root_lp import configure_feature_pass, features_sane
from offline_gnn.features_factor import extract_factor_features
```

To prepare a graph, build a fresh model, call `configure_feature_pass(model)`
from `offline_gnn/root_lp.py`, retain its returned event handler, run `model.optimize()`,
and call `extract_factor_features(model, builder)`. Check `features_sane`
before use. Obtain labels from a separate full solve, align them by variable
name, and record the solve status/gap rather than treating every incumbent as
optimal. Keep the same feature-pass configuration at inference.

Training saves `factor_gnn.pt` with architecture dimensions, weights and
training-set normalization. The network performs residual variable-to-factor
and factor-to-variable message passing. Region scores form a four-way softmax;
lane scores form a three-way softmax with a zero score for staying.
`structured_fixmap` converts confident complete decisions into binary
assignments for a freshly built reduced MINLP.


