"""Shared root-relaxation configuration for feature extraction.

Disable presolve, cuts and symmetry so original variable/constraint names
remain aligned. Interrupt after the first LP; use the same settings for
training features and inference."""
from __future__ import annotations

import numpy as np
from pyscipopt import Eventhdlr, SCIP_EVENTTYPE, SCIP_PARAMSETTING

# SCIP's infinity is 1e20; anything near it in a feature column means there was
# no usable LP solution to read, not a real value.
FEATURE_SANE_LIMIT = 1e6

# Presolve and separation must stay OFF: the bipartite graph is indexed by the
# ORIGINAL variables/constraints, so any presolve reduction would change the
# graph and break alignment with the training data.
#
# `misc/usesymmetry` belongs to that same list and was missed. Symmetry handling
# ALSO adds constraints and fixes variables, i.e. it changes exactly the graph
# this pass exists to read off unmodified. It was harmless while it stayed
# inactive, and the MINLP coupling is what exposed it: on a coupled model with
# presolve off, SCIP found 33 symmetry generators and cut the root node off
# before solving a single LP, reporting `infeasible` for a model that solves to
# optimality with presolve on. The feature pass then returned SCIP's infinity as
# the dual bound, the LP-derived feature columns were degenerate, and the GNN
# fixed the ego's lane change away (dObj +99.9 on a 141.7 objective).
#
# Turning it off is free for everything already recorded: on the uncoupled model
# the extracted features are BIT-IDENTICAL either way and the root dual bound is
# unchanged (134.5), so no checkpoint's train/inference consistency moves.
FEATURE_PASS_PARAMS = [
    ("display/verblevel", 0),
    ("presolving/maxrounds", 0),
    ("presolving/maxrestarts", 0),
    ("separating/maxrounds", 0),
    ("separating/maxroundsroot", 0),
    ("misc/usesymmetry", 0),
    ("limits/nodes", 1),
]


class StopAtFirstLP(Eventhdlr):
    """Interrupt the solve as soon as the first root LP has been solved."""

    def __init__(self):
        super().__init__()
        self.fired = 0

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, self)

    def eventexec(self, event):
        self.fired += 1
        self.model.interruptSolve()


def features_sane(feats: dict, limit: float = FEATURE_SANE_LIMIT) -> bool:
    """Did the feature pass actually produce an LP solution to read features off?

    When the root LP is infeasible, or the pass hits its time limit before any
    LP is solved, SCIP still answers `getVal` -- with its infinity (1e20). That
    leaks into the LP-derived columns (var 16-19 sol_val/at_lb/at_ub/frac,
    con 10-12 is_tight/slack/dual) and turns the sample into garbage: as
    TRAINING data it blows the loss up by ~8 orders of magnitude, and at
    INFERENCE time it is a graph the model has never seen, so its predictions
    are meaningless and must not be fixed into the MINLP.

    Seen on 5/200 intentionally-boxed instances (all of them the hardest,
    `timelimit` ones) and 0/299 rollout steps -- which is why it stayed hidden
    until targeted hard-instance collection started.
    """
    for key in ("variable_features", "constraint_features", "edge_features"):
        arr = feats.get(key)
        if arr is None:
            return False
        arr = np.asarray(arr)
        if not np.all(np.isfinite(arr)) or np.abs(arr).max(initial=0.0) > limit:
            return False
    return True


def configure_feature_pass(model, time_limit: float = 0.0, stop_at_first_lp: bool = True):
    """Apply the feature-pass parameters to `model`; return the handler (or None).

    `time_limit <= 0` means no limit. Keep the handler alive for as long as the
    model is being solved -- pyscipopt does not own the reference.
    """
    for name, val in FEATURE_PASS_PARAMS:
        model.setParam(name, val)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    if time_limit and time_limit > 0:
        model.setParam("limits/time", float(time_limit))
    hdlr = None
    if stop_at_first_lp:
        hdlr = StopAtFirstLP()
        model.includeEventhdlr(hdlr, "StopAtFirstLP",
                               "interrupt once the first root LP is solved")
    return hdlr
