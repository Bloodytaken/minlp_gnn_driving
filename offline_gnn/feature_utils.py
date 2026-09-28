"""Name classification and bound helpers for the MINLP graph features."""

import re
import math
import contextlib
import numpy as np


@contextlib.contextmanager
def _no_scip_output():
    """No-op on macOS: fd-level redirection causes segfaults with SCIP threads."""
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Regex patterns to classify MINLP variable names produced by MINLPBuilder
_VAR_PATTERNS = {
    "state":      re.compile(r"^x\["),         # ego state  x[nid, i]
    "control":    re.compile(r"^u\["),          # ego control u[nid, j]
    "lane_idx":   re.compile(r"^lane_idx\["),   # integer lane index
    "r_ref":      re.compile(r"^r_ref\["),      # reference radius
    "b_pos":      re.compile(r"^b_pos\["),      # lane-change binary (+)
    "b_neg":      re.compile(r"^b_neg\["),      # lane-change binary (-)
    "delta_lane": re.compile(r"^delta_lane\["), # lane change delta
    "slack":      re.compile(r"^slack_lat\["),  # lateral slack
    "gamma":      re.compile(r"^gamma_"),       # safety-region binary
    "obj":        re.compile(r"^t_obj"),        # objective epigraph
}

_CONS_PATTERNS = {
    "init":       re.compile(r"^x0\[|^lane_idx_root"),
    "dynamics":   re.compile(r"^dyn\["),
    "lane_rec":   re.compile(r"^lane_rec\[|^one_dir\[|^delta_lane_def\["),
    "dwell":      re.compile(r"^dwell_"),
    "bound":      re.compile(r"^u_min\[|^u_max\[|^vx_|^vy_"),
    "lane_bnd":   re.compile(r"^lane_upper\[|^lane_lower\[|^r_ref_affine\["),
    "safety":     re.compile(r"^region_sum\[|^front\[|^back\[|^inner\[|^outer\["),
    "obj":        re.compile(r"^obj_epigraph"),
}

_VAR_KEYS  = list(_VAR_PATTERNS.keys())   # fixed ordering for one-hot
_CONS_KEYS = list(_CONS_PATTERNS.keys())

# Normalisation constants for opponent-relative features
# (tree state ordering is [x, y, vx, vy]; MINLP ego state is [x, vx, y, vy])
_D_LONG = 50.0   # longitudinal separation scale [m]
_WL     =  3.5   # lane width — lateral separation scale [m]
_V_MAX  =  5.0   # relative speed scale [m/s]

_NID_RE = re.compile(r"\[(\d+)")


def _parse_node_id(name: str):
    """
    Extract the scenario-tree node id from a SCIP variable / constraint name.

    Naming convention (MINLPBuilder):
        x[{nid},{i}]  u[{nid},{j}]  lane_idx[{nid}]  gamma_f[{nid}]  …
    Returns the first integer inside brackets, or None for global names
    (t_obj, x0[i], lane_idx_root, obj_epigraph).
    """
    m = _NID_RE.search(name)
    return int(m.group(1)) if m else None


def _classify(name, patterns, keys):
    """Return a one-hot list identifying which pattern group `name` matches."""
    vec = [0.0] * len(keys)
    for idx, key in enumerate(keys):
        if patterns[key].search(name):
            vec[idx] = 1.0
            return vec
    return vec  # all zeros if unmatched


def _safe_lb(v, default=-1e20):
    lb = v.getLbLocal()
    return lb if lb > -1e20 else default


def _safe_ub(v, default=1e20):
    ub = v.getUbLocal()
    return ub if ub < 1e20 else default


def _var_key(v):
    """Return the *original* variable name, stripping SCIP's 't_' transform prefix."""
    name = v if isinstance(v, str) else v.name
    # After presolving SCIP renames vars: "x[0,0]" -> "t_x[0,0]"
    if name.startswith("t_"):
        return name[2:]
    return name


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
