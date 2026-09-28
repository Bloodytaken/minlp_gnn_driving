"""SCIP encoding of nearest-leader selection and the car-following acceleration bound.

The ego is a decision-dependent leader candidate; background paths are given.
At switching surfaces, adjacent branches can be feasible within SCIP tolerance."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


from offline_gnn.factor_schema import factor, add_linear, add_product, add_square

Interval = Tuple[float, float]

_M_PAD = 1.0
_S_FLOOR = 1e-2          # matches `find_leader`'s max(ds - L, 1e-2)


def _big_m(*needs: float) -> float:
    return float(max(0.0, max(needs))) + _M_PAD


def _rng(q) -> Interval:
    """Range of a candidate field given either a constant or (var, lo, hi)."""
    if isinstance(q, tuple):
        return float(q[1]), float(q[2])
    return float(q), float(q)


def _expr(q):
    return q[0] if isinstance(q, tuple) else float(q)


@dataclass(frozen=True)
class FollowParams:
    """The numbers `follow_bound` and `find_leader` read. Snapshotted so the
    encoding cannot follow a mutated params object."""

    a_bound: float = 4.0        # deceleration scale of the bound AND of s*
    a_max: float = 4.0          # control limit the result is clipped to
    s0: float = 2.0
    T: float = 1.0
    b: float = 2.0
    w_follow: float = 2.6
    look: float = 60.0
    vehicle_length: float = 4.5

    @classmethod
    def from_gap_params(cls, gp, a_max: float) -> "FollowParams":
        return cls(a_bound=float(gp.a_bound if gp.a_bound is not None else a_max),
                   a_max=float(a_max), s0=float(gp.s0), T=float(gp.T),
                   b=float(gp.b), w_follow=float(gp.follow_halfwidth()),
                   look=float(gp.look),
                   vehicle_length=float(gp.vehicle_length))


@dataclass
class Candidate:
    """One vehicle the opponent might be following.

    `x`, `y`, `v` are each either a float (a background vehicle on its given
    trajectory) or a `(var, lo, hi)` triple (the ego, whose position is being
    decided).
    """

    name: str
    x: Any
    y: Any
    v: Any


def prune_candidates(cands: Sequence[Candidate], *, xo_rng: Interval,
                     yo: float, p: FollowParams) -> Tuple[List[Candidate], List[str]]:
    """Drop candidates that provably cannot qualify. Returns (kept, dropped)."""
    kept, dropped = [], []
    for c in cands:
        x_lo, x_hi = _rng(c.x)
        y_lo, y_hi = _rng(c.y)
        ds_lo, ds_hi = x_lo - xo_rng[1], x_hi - xo_rng[0]
        dn_lo, dn_hi = y_lo - yo, y_hi - yo
        if ds_hi <= 0.0:                       # never ahead
            dropped.append(f"{c.name}:behind")
        elif ds_lo > p.look:                   # never within look
            dropped.append(f"{c.name}:far")
        elif dn_lo >= p.w_follow or dn_hi <= -p.w_follow:
            dropped.append(f"{c.name}:lateral")
        else:
            kept.append(c)
    return kept, dropped


def _indicator_ahead(model, ds, ds_rng: Interval, *, name: str, stats):
    """f == 1 <=> ds > 0, decided by bounds where possible."""
    lo, hi = ds_rng
    if lo > 0.0:
        return 1.0
    if hi <= 0.0:
        return 0.0
    f = model.addVar(vtype="B", name=f"lead_ahead[{name}]")
    M = _big_m(abs(lo), abs(hi))
    # f == 1 -> ds >= 0 ; f == 0 -> ds <= 0
    model.addCons(ds >= -M * (1 - f), name=f"ahead_pos[{name}]")
    model.addCons(ds <= M * f, name=f"ahead_neg[{name}]")
    stats["n_bin"] += 1
    stats["n_rows"] += 2
    return f


def _indicator_lateral(model, dn, dn_rng: Interval, *, w: float, name: str, stats):
    """l == 1 <=> |dn| < w, as the AND of two one-sided indicators."""
    lo, hi = dn_rng
    if lo > -w and hi < w:
        return 1.0
    if lo >= w or hi <= -w:
        return 0.0
    l = model.addVar(vtype="B", name=f"lead_lat[{name}]")
    M = _big_m(abs(lo) + w, abs(hi) + w)
    # l == 1 -> -w <= dn <= w   (the two rows below)
    model.addCons(dn <= w + M * (1 - l), name=f"lat_hi[{name}]")
    model.addCons(dn >= -w - M * (1 - l), name=f"lat_lo[{name}]")
    # l == 0 -> |dn| >= w, via a side binary so the disjunction is exact
    sgn = model.addVar(vtype="B", name=f"lead_side[{name}]")
    model.addCons(dn >= w - M * (l + (1 - sgn)), name=f"lat_out_hi[{name}]")
    model.addCons(dn <= -w + M * (l + sgn), name=f"lat_out_lo[{name}]")
    stats["n_bin"] += 2
    stats["n_rows"] += 4
    return l


def _is_const(q, val=None):
    if not isinstance(q, float):
        return False
    return True if val is None else q == val


def _and(model, a, b, *, name: str, stats):
    if _is_const(a, 1.0):
        return b
    if _is_const(b, 1.0):
        return a
    if _is_const(a, 0.0) or _is_const(b, 0.0):
        return 0.0
    q = model.addVar(vtype="B", name=f"lead_ok[{name}]")
    model.addCons(q <= a, name=f"and_a[{name}]")
    model.addCons(q <= b, name=f"and_b[{name}]")
    model.addCons(q >= a + b - 1, name=f"and_c[{name}]")
    stats["n_bin"] += 1
    stats["n_rows"] += 3
    return q


def _bound_value(model, s_expr, s_rng: Interval, vo, vo_rng: Interval,
                 vj, vj_rng: Interval, *, p: FollowParams, name: str, stats, factor_specs):
    """a_follow_j = a_b (1 - (s*/s)^2) for one candidate, as an expression.

    Two nonlinear equalities, both authored here so their gradients are known:
    the headway product `v * dv` and the quotient `qr * s == s*`. Everything
    else is linear. `max(0, head)` is skipped whenever its bounds settle it,
    which on this scenario is most of the time -- a following vehicle is nearly
    always slower than the desired headway term's zero crossing.
    """
    a_b = p.a_bound
    k = 2.0 * math.sqrt(max(a_b * p.b, 1e-9))

    # head = v*T + v*dv/k  with dv = vo - vj
    dv_lo, dv_hi = vo_rng[0] - vj_rng[1], vo_rng[1] - vj_rng[0]
    prod_lo = min(vo_rng[0] * dv_lo, vo_rng[0] * dv_hi,
                  vo_rng[1] * dv_lo, vo_rng[1] * dv_hi)
    prod_hi = max(vo_rng[0] * dv_lo, vo_rng[0] * dv_hi,
                  vo_rng[1] * dv_lo, vo_rng[1] * dv_hi)
    head_lo = vo_rng[0] * p.T + prod_lo / k
    head_hi = vo_rng[1] * p.T + prod_hi / k

    if head_lo >= 0.0:                      # max(0, .) is the identity
        sstar_lo, sstar_hi = p.s0 + head_lo, p.s0 + head_hi
        sstar = model.addVar(lb=sstar_lo, ub=sstar_hi, name=f"sstar[{name}]")
        model.addCons(sstar == p.s0 + vo * p.T + vo * (vo - vj) / k,
                      name=f"sstar_def[{name}]")
        fs = factor("opponent_belief_dynamics", "eq", constant=-p.s0, output=sstar)
        add_linear(fs, sstar); add_linear(fs, vo, -p.T)
        add_square(fs, vo, -1/k); add_product(fs, vo, vj, 1/k)
        factor_specs[f"sstar_def[{name}]"] = fs
        stats["n_nonlinear"] += 1
        stats["n_rows"] += 1
    elif head_hi <= 0.0:                    # headway never positive
        sstar, sstar_lo, sstar_hi = float(p.s0), p.s0, p.s0
    else:
        raw_lo, raw_hi = head_lo, head_hi
        raw = model.addVar(lb=raw_lo, ub=raw_hi, name=f"head_raw[{name}]")
        model.addCons(raw == vo * p.T + vo * (vo - vj) / k,
                      name=f"head_def[{name}]")
        fs = factor("opponent_belief_dynamics", "eq", output=raw)
        add_linear(fs, raw); add_linear(fs, vo, -p.T)
        add_square(fs, vo, -1/k); add_product(fs, vo, vj, 1/k)
        factor_specs[f"head_def[{name}]"] = fs
        stats["n_nonlinear"] += 1
        pos = model.addVar(lb=0.0, ub=max(0.0, raw_hi), name=f"head_pos[{name}]")
        bpos = model.addVar(vtype="B", name=f"head_b[{name}]")
        M = _big_m(abs(raw_lo), abs(raw_hi))
        model.addCons(pos >= raw, name=f"head_ge[{name}]")
        model.addCons(pos <= raw + M * (1 - bpos), name=f"head_le[{name}]")
        model.addCons(pos <= M * bpos, name=f"head_zero[{name}]")
        sstar_lo, sstar_hi = p.s0, p.s0 + max(0.0, raw_hi)
        sstar = model.addVar(lb=sstar_lo, ub=sstar_hi, name=f"sstar[{name}]")
        model.addCons(sstar == p.s0 + pos, name=f"sstar_def[{name}]")
        stats["n_bin"] += 1
        stats["n_rows"] += 5

    # Saturate the ratio where the FINAL acceleration reaches -a_max.
    # min(clipped theta, max(-a_max, follow)) equals clip(min(theta, follow)).
    # This avoids huge negative bounds / big-M values at centimetre gaps.
    qcap = math.sqrt(1 + p.a_max/a_b)
    s_lo, s_hi = max(s_rng[0], _S_FLOOR), max(s_rng[1], _S_FLOOR)
    qr_lo, qr_hi = min(qcap, sstar_lo/s_hi), min(qcap, sstar_hi/s_lo)
    qr = model.addVar(lb=qr_lo, ub=qr_hi, name=f"qratio[{name}]")
    if sstar_hi/s_lo <= qcap:
        model.addCons(qr*s_expr == sstar, name=f"qratio_def[{name}]")
        fs = factor("opponent_belief_dynamics", "eq")
        add_product(fs, qr, s_expr); add_linear(fs, sstar, -1)
        factor_specs[f"qratio_def[{name}]"] = fs
        stats["n_rows"] += 1; stats["n_nonlinear"] += 1
    elif sstar_lo/s_hi < qcap:
        sat = model.addVar(vtype="B", name=f"qratio_sat[{name}]")
        M = _big_m(sstar_hi)
        model.addCons(qr*s_expr <= sstar, name=f"qratio_hi[{name}]")
        model.addCons(qr*s_expr >= sstar-M*sat, name=f"qratio_lo[{name}]")
        model.addCons(qr >= qcap*sat, name=f"qratio_cap[{name}]")
        for suffix, sense in (("hi", "le"), ("lo", "ge")):
            fs = factor("opponent_belief_dynamics", sense)
            add_product(fs, qr, s_expr); add_linear(fs, sstar, -1)
            if suffix == "lo": add_linear(fs, sat, M)
            factor_specs[f"qratio_{suffix}[{name}]"] = fs
        stats["n_bin"] += 1; stats["n_rows"] += 3; stats["n_nonlinear"] += 2
    a_lo, a_hi = max(-p.a_max, a_b*(1-qr_hi**2)), a_b*(1-qr_lo**2)
    af = model.addVar(lb=a_lo, ub=max(a_lo, a_hi), name=f"afollow[{name}]")
    model.addCons(af == a_b*(1-qr*qr), name=f"afollow_def[{name}]")
    fs = factor("opponent_belief_dynamics", "eq", constant=-a_b, output=af)
    add_linear(fs, af); add_square(fs, qr, a_b)
    factor_specs[f"afollow_def[{name}]"] = fs
    stats["n_nonlinear"] += 1; stats["n_rows"] += 1
    return af, (a_lo, max(a_lo, a_hi))


def encode_follow(model, *, xo, xo_rng: Interval, yo: float, vo, vo_rng: Interval,
                  candidates: Sequence[Candidate], p: FollowParams, name: str,
                  factor_specs=None):
    """a_follow for one opponent at one depth, with the leader chosen inside
    the model. Returns `(a_follow, (lo, hi), stats)`; `a_follow` is a float
    `p.a_max` when no candidate survives pruning, i.e. free driving."""
    factor_specs = {} if factor_specs is None else factor_specs
    stats = {"n_bin": 0, "n_rows": 0, "n_nonlinear": 0, "dropped": [],
             "n_cand": 0}
    kept, dropped = prune_candidates(candidates, xo_rng=xo_rng, yo=yo, p=p)
    stats["dropped"] = dropped
    stats["n_cand"] = len(kept)
    if not kept:
        return float(p.a_max), (p.a_max, p.a_max), stats

    qs, dss, afs, af_rngs = [], [], [], []
    for c in kept:
        x_lo, x_hi = _rng(c.x)
        y_lo, y_hi = _rng(c.y)
        v_lo, v_hi = _rng(c.v)
        ds = _expr(c.x) - xo
        ds_rng = (x_lo - xo_rng[1], x_hi - xo_rng[0])
        dn = _expr(c.y) - yo
        dn_rng = (y_lo - yo, y_hi - yo)
        f = _indicator_ahead(model, ds, ds_rng, name=f"{name}/{c.name}", stats=stats)
        l = _indicator_lateral(model, dn, dn_rng, w=p.w_follow,
                               name=f"{name}/{c.name}", stats=stats)
        near = _indicator_ahead(model, p.look-ds,
                (p.look-ds_rng[1], p.look-ds_rng[0]),
                name=f"{name}/{c.name}/look", stats=stats) if ds_rng[1] > p.look else 1.0
        fl = _and(model, f, l, name=f"{name}/{c.name}/path", stats=stats)
        q = _and(model, fl, near, name=f"{name}/{c.name}", stats=stats)
        # bumper-to-bumper gap, floored exactly as `find_leader` floors it
        s_rng = (max(ds_rng[0] - p.vehicle_length, _S_FLOOR),
                 max(ds_rng[1] - p.vehicle_length, _S_FLOOR))
        # s == max(ds - L, floor), pinned from BOTH sides. Bounding it only
        # from below would leave the solver free to inflate the gap and weaken
        # the bound on itself -- the encoding has to reproduce the law, not
        # merely relax it, and with no objective term on `s` nothing else would
        # stop it.
        s_var = model.addVar(lb=s_rng[0], ub=s_rng[1], name=f"sgap[{name}/{c.name}]")
        if ds_rng[0] - p.vehicle_length >= _S_FLOOR:
            model.addCons(s_var == ds - p.vehicle_length,
                          name=f"sgap_def[{name}/{c.name}]")
            stats["n_rows"] += 1
        else:
            bs = model.addVar(vtype="B", name=f"sgap_b[{name}/{c.name}]")
            Ms = _big_m(abs(ds_rng[0] - p.vehicle_length - _S_FLOOR),
                        abs(ds_rng[1] - p.vehicle_length - _S_FLOOR))
            model.addCons(s_var >= ds - p.vehicle_length, name=f"sgap_a[{name}/{c.name}]")
            model.addCons(s_var >= _S_FLOOR, name=f"sgap_b1[{name}/{c.name}]")
            model.addCons(s_var <= ds - p.vehicle_length + Ms * (1 - bs),
                          name=f"sgap_c[{name}/{c.name}]")
            model.addCons(s_var <= _S_FLOOR + Ms * bs, name=f"sgap_d[{name}/{c.name}]")
            stats["n_bin"] += 1
            stats["n_rows"] += 4
        af, af_rng = _bound_value(model, s_var, s_rng, vo, vo_rng,
                                  _expr(c.v), (v_lo, v_hi), p=p,
                                  name=f"{name}/{c.name}", stats=stats, factor_specs=factor_specs)
        qs.append(q); dss.append((ds, ds_rng)); afs.append(af); af_rngs.append(af_rng)

    # ---- selection: the nearest qualifying candidate ----
    sels = []
    for c, q in zip(kept, qs):
        sel = model.addVar(vtype="B", name=f"sel[{name}/{c.name}]")
        model.addCons(sel <= q, name=f"sel_q[{name}/{c.name}]")
        sels.append(sel)
        stats["n_bin"] += 1
        stats["n_rows"] += 1
    model.addCons(sum(sels) <= 1, name=f"sel_one[{name}]")
    stats["n_rows"] += 1
    # something qualifies -> something is selected
    for c, q in zip(kept, qs):
        model.addCons(sum(sels) >= q, name=f"sel_force[{name}/{c.name}]")
        stats["n_rows"] += 1
    # argmin: sel_j and q_k  ->  ds_j <= ds_k
    for j, (cj, selj) in enumerate(zip(kept, sels)):
        dsj, rj = dss[j]
        for k, (ck, qk) in enumerate(zip(kept, qs)):
            if j == k:
                continue
            dsk, rk = dss[k]
            M = _big_m(abs(rj[1] - rk[0]), abs(rj[0] - rk[1]))
            model.addCons(dsj <= dsk + M * (1 - selj) + M * (1 - qk),
                          name=f"argmin[{name}/{cj.name}->{ck.name}]")
            stats["n_rows"] += 1

    # ---- a_follow = sum_j sel_j * af_j + (1 - sum sel_j) * a_max ----
    lo = min(r[0] for r in af_rngs + [(p.a_max, p.a_max)])
    hi = max(r[1] for r in af_rngs + [(p.a_max, p.a_max)])
    terms = []
    for j, (sel, af, r) in enumerate(zip(sels, afs, af_rngs)):
        z = model.addVar(lb=min(0.0, r[0]), ub=max(0.0, r[1]),
                         name=f"pick[{name}/{kept[j].name}]")
        M_lo, M_hi = r
        model.addCons(z <= M_hi * sel, name=f"pick_ub[{name}/{j}]")
        model.addCons(z >= M_lo * sel, name=f"pick_lb[{name}/{j}]")
        model.addCons(z <= af - M_lo * (1 - sel), name=f"pick_a[{name}/{j}]")
        model.addCons(z >= af - M_hi * (1 - sel), name=f"pick_b[{name}/{j}]")
        stats["n_rows"] += 4
        terms.append(z)
    a_follow = model.addVar(lb=lo, ub=hi, name=f"a_follow[{name}]")
    model.addCons(a_follow == sum(terms) + p.a_max * (1 - sum(sels)),
                  name=f"a_follow_def[{name}]")
    stats["n_rows"] += 1
    return a_follow, (lo, hi), stats


def encode_min(model, a_theta, at_rng: Interval, a_follow, af_rng: Interval,
               *, name: str):
    """a == min(a_theta, a_follow), EXACTLY.

    The two `<=` rows alone would only bound `a` from above by both; a solver
    minimising something that prefers small accelerations would then be free to
    return anything below the smaller one. The two big-M rows force equality
    with whichever side is actually smaller.
    """
    stats = {"n_bin": 0, "n_rows": 0}
    if isinstance(a_theta, float) and isinstance(a_follow, float):
        return float(min(a_theta, a_follow)), (min(a_theta, a_follow),) * 2, stats
    if at_rng[1] <= af_rng[0]:
        return a_theta, at_rng, stats
    if af_rng[1] <= at_rng[0]:
        return a_follow, af_rng, stats
    lo = min(at_rng[0], af_rng[0])
    hi = min(at_rng[1], af_rng[1])
    a = model.addVar(lb=lo, ub=hi, name=f"a_min[{name}]")
    d = model.addVar(vtype="B", name=f"min_b[{name}]")
    M = _big_m(abs(at_rng[1] - af_rng[0]), abs(af_rng[1] - at_rng[0]))
    model.addCons(a <= a_theta, name=f"min_a[{name}]")
    model.addCons(a <= a_follow, name=f"min_b2[{name}]")
    model.addCons(a >= a_theta - M * d, name=f"min_c[{name}]")
    model.addCons(a >= a_follow - M * (1 - d), name=f"min_d[{name}]")
    stats["n_bin"] += 1
    stats["n_rows"] += 4
    return a, (lo, hi), stats
