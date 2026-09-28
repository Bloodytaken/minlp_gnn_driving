"""SCIP encoding of the interaction gate, type-dependent target speed and acceleration clipping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


from parameters import OpponentModelParams

# Same mapping `OpponentModel.make_mu_functions` uses.
THETA_SIGN = {"cau": -1.0, "agg": +1.0}

# Slack added on top of every big-M so a coefficient that is *exactly* tight
# cannot be defeated by floating-point noise.
_M_PAD = 1.0


# ---------------------------------------------------------------------------
# Parameters and interval arithmetic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LawParams:
    """The five numbers the reactive law actually reads, snapshotted so the
    encoding cannot silently follow a mutated params object."""

    d_int: float
    k_c: float
    delta_v: float
    v_star: float
    a_max: float

    @classmethod
    def from_params(cls, p: OpponentModelParams) -> "LawParams":
        return cls(d_int=float(p.d_int), k_c=float(p.k_c),
                   delta_v=float(p.delta_v), v_star=float(p.v_star),
                   a_max=float(p.a_max))

    @classmethod
    def from_model(cls, om) -> "LawParams":
        return cls.from_params(om.params)


Interval = Tuple[float, float]


def sq_interval(lo: float, hi: float) -> Interval:
    """Range of t**2 for t in [lo, hi]. Zero is attainable only if the interval
    straddles it -- otherwise the minimum sits at the endpoint nearest zero."""
    if lo <= 0.0 <= hi:
        return 0.0, max(lo * lo, hi * hi)
    return min(lo * lo, hi * hi), max(lo * lo, hi * hi)


def clip_interval(lo: float, hi: float, a_max: float) -> Interval:
    return max(lo, -a_max), min(hi, a_max)


def _big_m(*needs: float) -> float:
    """Smallest dominating big-M for the listed required magnitudes."""
    return float(max(0.0, max(needs))) + _M_PAD


# ---------------------------------------------------------------------------
# Symbolic encoders
# ---------------------------------------------------------------------------

def encode_gate(model, xe, ye, xo, yo_const, *, xe_rng: Interval, ye_rng: Interval,
                xo_rng: Interval, p: LawParams, name: str):
    """Interaction indicator z == 1  <=>  ||p_e - p_o|| <= d_int.

    `xe`/`ye` are the ego position variables, `xo` the opponent longitudinal
    variable, `yo_const` its (deterministic) lateral position. Returns
    `(z, stats)` where z is a binary Var, or a float 0.0/1.0 when the geometry
    bounds settle the question without any search.

    The squared distance is introduced as its own bounded variable so the ONLY
    nonconvex row in the whole model is its defining equality; both big-M rows
    are then linear in it.
    """
    stats = {"n_bin": 0, "n_nonlinear": 0, "n_rows": 0, "reduced": None}

    dx_lo, dx_hi = xe_rng[0] - xo_rng[1], xe_rng[1] - xo_rng[0]
    dy_lo, dy_hi = ye_rng[0] - yo_const, ye_rng[1] - yo_const
    sx_lo, sx_hi = sq_interval(dx_lo, dx_hi)
    sy_lo, sy_hi = sq_interval(dy_lo, dy_hi)
    g2_lo, g2_hi = sx_lo + sy_lo, sx_hi + sy_hi
    thr = p.d_int * p.d_int

    # Bounds alone can decide the switch -- the common case for a phantom or a
    # far-away focus, and it removes the nonconvex row entirely.
    if g2_hi <= thr:
        stats["reduced"] = "always_in"
        return 1.0, stats
    if g2_lo >= thr:
        stats["reduced"] = "always_out"
        return 0.0, stats

    g2 = model.addVar(lb=g2_lo, ub=g2_hi, name=f"g2[{name}]")
    model.addCons(g2 == (xe - xo) * (xe - xo)
                  + (ye - yo_const) * (ye - yo_const),
                  name=f"gate_def[{name}]")
    stats["n_nonlinear"] += 1
    stats["n_rows"] += 1
    # Hand the feature extractor what it cannot read back off SCIP. `getValsLinear`
    # raises on a nonlinear row, so the base extractor drops it -- and this is the one
    # row that says what g2 MEANS. Without it g2 is an isolated node in the
    # bipartite graph (measured: degree 0) and the geometry driving z is invisible.
    # We author the row, so its gradient is known in closed form; the extractor
    # linearises it at the LP point. See offline_gnn.features_factor._linearize_sqdist.
    #
    # This spec is a tuple of NAMES, deliberately: it describes the row for the
    # GNN's graph and nothing else. The solver goes on solving the exact
    # nonconvex equality added just above. The linearisation is an OVER-estimator
    # of a concave function and would cut off feasible points if it were ever
    # added as a relaxation -- do not repurpose it as one without first proving
    # validity (it is not valid as written).
    stats["nl_row"] = ("sqdist", g2.name, xe.name, xo.name, ye.name, float(yo_const))

    z = model.addVar(vtype="B", name=f"z_int[{name}]")
    stats["n_bin"] += 1
    Mg = _big_m(g2_hi - thr, thr - g2_lo)
    # z == 1  ->  g2 <= d_int^2 ;  z == 0  ->  g2 >= d_int^2
    model.addCons(g2 <= thr + Mg * (1 - z), name=f"gate_in[{name}]")
    model.addCons(g2 >= thr - Mg * z, name=f"gate_out[{name}]")
    stats["n_rows"] += 2
    return z, stats


def encode_vref(model, z, vxe, *, theta_sign: float, vxe_rng: Interval,
                p: LawParams, name: str):
    """v_ref = z*(vx_e + theta*delta_v) + (1-z)*v_star.

    Exact big-M linearisation of the bilinear z*vx_e (exact because z is
    binary). When `z` arrives as a resolved constant the whole thing collapses
    to a linear expression and no variable is created at all.
    """
    stats = {"n_bin": 0, "n_rows": 0, "reduced": None}
    on_lo = vxe_rng[0] + theta_sign * p.delta_v
    on_hi = vxe_rng[1] + theta_sign * p.delta_v

    if isinstance(z, float):
        stats["reduced"] = "gate_const"
        if z >= 0.5:
            return vxe + theta_sign * p.delta_v, (on_lo, on_hi), stats
        return p.v_star, (p.v_star, p.v_star), stats

    lo, hi = min(on_lo, p.v_star), max(on_hi, p.v_star)
    vref = model.addVar(lb=lo, ub=hi, name=f"vref[{name}]")
    Mv = _big_m(hi - min(on_lo, p.v_star), max(on_hi, p.v_star) - lo)
    model.addCons(vref >= vxe + theta_sign * p.delta_v - Mv * (1 - z),
                  name=f"vref_on_lo[{name}]")
    model.addCons(vref <= vxe + theta_sign * p.delta_v + Mv * (1 - z),
                  name=f"vref_on_hi[{name}]")
    model.addCons(vref >= p.v_star - Mv * z, name=f"vref_off_lo[{name}]")
    model.addCons(vref <= p.v_star + Mv * z, name=f"vref_off_hi[{name}]")
    stats["n_rows"] += 4
    return vref, (lo, hi), stats


def encode_clip(model, araw, araw_rng: Interval, *, a_max: float, name: str):
    """ac == clip(araw, -a_max, +a_max), as min then max.

    Interval arithmetic settles the piece outright whenever it can, which is
    worth doing: each avoided encoding is 2 binaries and 8 rows, and on a real
    instance a large share of the (parent, tag) groups are decided this way.
    """
    stats = {"n_bin": 0, "n_rows": 0, "reduced": None}
    lo, hi = araw_rng

    if lo >= -a_max and hi <= a_max:            # never saturates
        stats["reduced"] = "never"
        return araw, (lo, hi), stats
    if lo >= a_max:                             # pinned at the ceiling
        stats["reduced"] = "always_hi"
        return float(a_max), (a_max, a_max), stats
    if hi <= -a_max:                            # pinned at the floor
        stats["reduced"] = "always_lo"
        return float(-a_max), (-a_max, -a_max), stats

    # w == min(a_max, araw)
    w_lo, w_hi = min(lo, a_max), min(hi, a_max)
    w = model.addVar(lb=w_lo, ub=w_hi, name=f"w_min[{name}]")
    b_hi = model.addVar(vtype="B", name=f"clip_hi[{name}]")
    M1 = _big_m(abs(a_max - lo), abs(a_max - hi))
    model.addCons(w <= a_max, name=f"clip_min_a[{name}]")
    model.addCons(w <= araw, name=f"clip_min_b[{name}]")
    model.addCons(w >= a_max - M1 * (1 - b_hi), name=f"clip_min_ca[{name}]")
    model.addCons(w >= araw - M1 * b_hi, name=f"clip_min_cb[{name}]")

    # ac == max(-a_max, w)
    ac_lo, ac_hi = max(w_lo, -a_max), max(w_hi, -a_max)
    ac = model.addVar(lb=ac_lo, ub=ac_hi, name=f"ac[{name}]")
    b_lo = model.addVar(vtype="B", name=f"clip_lo[{name}]")
    M2 = _big_m(abs(-a_max - w_lo), abs(-a_max - w_hi))
    model.addCons(ac >= -a_max, name=f"clip_max_a[{name}]")
    model.addCons(ac >= w, name=f"clip_max_b[{name}]")
    model.addCons(ac <= -a_max + M2 * (1 - b_lo), name=f"clip_max_ca[{name}]")
    model.addCons(ac <= w + M2 * b_lo, name=f"clip_max_cb[{name}]")

    stats["n_bin"] += 2
    stats["n_rows"] += 8
    return ac, (ac_lo, ac_hi), stats
