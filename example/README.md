# Example: Going through the pipeline.

Run from `mi_dual_mpc/MINLP/` with the dependencies in `requirements.txt`:

```bash
python -m example.run_pipeline
```

On the current machine, the tested environment is:

```bash
conda run -n nmpc python -m example.run_pipeline
```

The example runs these steps using the existing formulation and training code:

1. Generate 10 training and 2 validation problems with different random seeds.
2. Extract each root-relaxation graph and solve a separate full MINLP to
   optimality. Match labels to graph variables by name.
3. Train the GNN for 40 epochs and reload the selected checkpoint, including
   normalization computed from the training set.
4. Predict decisions for a new problem (seed 2000), fix groups with confidence
   at least 0.95, and solve the reduced MINLP. Compare against a separate full
   solve; its solution is used only for evaluation, not for prediction/fixing.

The scene has three lanes and one focus opponent ahead of the ego, with
`H=2`, `Hb=1`, and `dt=0.2 s`. This is a small pipeline check, with one held-out
planning step; it is not a closed-loop simulation or a paper benchmark.
The training, validation and test seeds are disjoint.

## Outputs

Generated files are written to `example/output/` (ignored by Git):

| File | Content |
| --- | --- |
| `dataset.pkl` | Graphs, aligned optimal labels, seeds and solve metadata |
| `factor_gnn.pt` | Trained weights, dimensions, normalization and thresholds |
| `training.log` | Training progress and validation metrics |
| `report.json` | Predictions, fixed variables, objectives, controls and timings |

Change the run length or output location with:

```bash
python -m example.run_pipeline --epochs 40 --time-limit 15 --output example/output
```

The script stops with an error if labels are not optimal, graph extraction is
incomplete, or the reduced solve fails. If no decisions pass the thresholds,
all candidates remain free (fix rate 0). It does not silently substitute the full solution for GNN predictions.

## Confidence thresholds and fix rate

Confidence is the largest **group softmax probability**, regions use four scores (front/back/left/right); lane
changes (stay/left/right). A group is fixed when `confidence >= threshold`.
Fixing a region assigns all four binaries; fixing a lane action assigns both
lane-change binaries, including two zeros for staying.

| Inference option | Default | Applies to |
| --- | --- | --- |
| `--gamma-threshold` | `0.95` | Each complete safe-region decision group |
| `--lane-threshold` | `0.95` | Future lane decisions (non-root nodes) |



Lower thresholds admit less confident predictions, which can worsen the
objective or make the reduced problem infeasible. A softmax confidence of
0.95 is not a guarantee of 95% correctness. Choose thresholds on validation
problems, checking fixing errors, feasibility, objective differences and total
runtime; a higher fix rate alone does not establish better performance.

From `MINLP/`, for example:

```bash
python -m example.run_pipeline \
  --gamma-threshold 0.95 --lane-threshold 0.80 --root-lane-threshold 1.01
```

These example flags override **inference only**, after training/checkpoint
selection; leaving an option unset uses the checkpoint's stored value. Each
invocation still runs the complete pipeline. 


For the verified epoch-5 checkpoint and the same held-out graph, threshold-only
re-decoding gives:

| Region | Fixed Rate |
| ---  | --- |
| 0.99 | 0 (0%) |
| 0.95 | 16 (72.7%) |
| 0.95 | 20 (90.9%) |
| 0.95 | 22 (100%) |


