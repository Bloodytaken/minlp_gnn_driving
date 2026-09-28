"""Belief Update in the Scenario Tree"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from pyscipopt import exp as scip_exp, quicksum

from offline_gnn.factor_schema import (add_exp_neg_square, add_linear, add_product,
                           factor)


THETAS = ("cau", "agg")


def _is_const(v) -> bool:
    return isinstance(v, (int, float))


def _e(v):
    return float(v) if _is_const(v) else v


def encode_belief_tree(model, tree, ac: Dict[Tuple[int, str], object], *,
                       sigma: float, Hb: int, root_belief: Dict[str, float],
                       factor_specs: Optional[Dict[str, dict]] = None):
    """Belief + weight variables for the whole tree.

    `ac[(parent_id, tag)]` are the clipped accelerations -- Var where the
    coupling is live, float where interval arithmetic already settled it. A
    parent whose two `ac` are both floats has a constant `d`, hence a constant
    belief update, and this emits no variables for it at all: it degrades
    cleanly back to the numbers the tree already carries.

    Returns `(w, b, stats)` with
        w[nid]          -> Var or float, the node weight
        b[(nid, theta)] -> Var or float, the belief at that node
    """
    var = sigma * sigma
    st = {"n_exp": 0, "n_var": 0, "n_rows": 0, "n_bilinear": 0,
          "parents_live": 0, "parents_const": 0}

    w: Dict[int, object] = {tree.root: 1.0}
    b: Dict[Tuple[int, str], object] = {(tree.root, t): float(root_belief[t])
                                        for t in THETAS}

    kids_of: Dict[int, List] = {}
    for n in tree.traverse():
        if n.parent is not None:
            kids_of.setdefault(n.parent, []).append(n)

    for p in tree.traverse():
        kids = kids_of.get(p.id)
        if not kids:
            continue

        a_cau, a_agg = ac.get((p.id, "cau")), ac.get((p.id, "agg"))
        if a_cau is None or a_agg is None:
            raise KeyError(f"no hypothesis acceleration for parent {p.id}; the "
                           f"belief encoding needs both hypotheses")

        # q = exp(-d^2 / 2 sigma^2), the single scalar the update turns on
        if _is_const(a_cau) and _is_const(a_agg):
            d_val = float(a_agg) - float(a_cau)
            q = math.exp(-0.5 * d_val * d_val / var)
            st["parents_const"] += 1
        else:
            d = model.addVar(lb=-1e3, ub=1e3, name=f"bel_d[{p.id}]")
            model.addCons(d == _e(a_agg) - _e(a_cau), name=f"bel_d_def[{p.id}]")
            q = model.addVar(lb=0.0, ub=1.0, name=f"bel_q[{p.id}]")
            q_name = f"bel_q_def[{p.id}]"
            model.addCons(q == scip_exp(-0.5 * (d * d) / var), name=q_name)
            if factor_specs is not None:
                fs = factor("belief_likelihood", "eq", output=q)
                add_linear(fs, q, 1.0, role="output")
                add_exp_neg_square(fs, d, -0.5 / var, -1.0)
                factor_specs[q_name] = fs
            st["n_exp"] += 1
            st["n_var"] += 2
            st["n_rows"] += 2
            st["parents_live"] += 1

        bp = {t: b[(p.id, t)] for t in THETAS}
        wp = w[p.id]
        all_const = _is_const(q) and all(_is_const(bp[t]) for t in THETAS) \
            and _is_const(wp)

        for k in kids:
            t = k.tag
            other = "agg" if t == "cau" else "cau"
            # evidence: own hypothesis contributes 1, the other contributes q
            if all_const:
                ev = float(bp[t]) + float(bp[other]) * float(q)
                w[k.id] = (float(wp) * float(bp[t])
                           if p.depth < Hb else float(wp))
                b[(k.id, t)] = float(bp[t]) / ev
                b[(k.id, other)] = float(bp[other]) * float(q) / ev
                continue

            ev_v = model.addVar(lb=1e-12, ub=2.0, name=f"bel_E[{k.id}]")
            ev_name = f"bel_E_def[{k.id}]"
            model.addCons(ev_v == _e(bp[t]) + _e(bp[other]) * _e(q), name=ev_name)
            if factor_specs is not None:
                fs = factor("belief_evidence", "eq", output=ev_v)
                add_linear(fs, ev_v, 1.0, role="output")
                add_linear(fs, bp[t], -1.0)
                add_product(fs, bp[other], q, -1.0)
                factor_specs[ev_name] = fs
            st["n_var"] += 1
            st["n_rows"] += 1
            if not (_is_const(bp[other]) or _is_const(q)):
                st["n_bilinear"] += 1

            if p.depth < Hb:
                w_v = model.addVar(lb=0.0, ub=1.0, name=f"bel_w[{k.id}]")
                w_name = f"bel_w_def[{k.id}]"
                model.addCons(w_v == _e(wp) * _e(bp[t]), name=w_name)
                if factor_specs is not None:
                    fs = factor("belief_weight", "eq", output=w_v)
                    add_linear(fs, w_v, 1.0, role="output")
                    add_product(fs, wp, bp[t], -1.0)
                    factor_specs[w_name] = fs
                w[k.id] = w_v
                st["n_var"] += 1
                st["n_rows"] += 1
                if not (_is_const(wp) or _is_const(bp[t])):
                    st["n_bilinear"] += 1
            else:
                # The reference stops branching, so probability mass is fixed,
                # while its belief state continues to update.
                w[k.id] = wp

            # b_child(t) * E = b_p(t) ;  b_child(other) * E = b_p(other) * q
            b_t = model.addVar(lb=0.0, ub=1.0, name=f"bel_b[{k.id},{t}]")
            bt_name = f"bel_b_def[{k.id},{t}]"
            model.addCons(b_t * ev_v == _e(bp[t]), name=bt_name)
            if factor_specs is not None:
                fs = factor("belief_posterior", "eq", output=b_t)
                add_product(fs, b_t, ev_v, 1.0,
                            roles=("output", "evidence"))
                add_linear(fs, bp[t], -1.0)
                factor_specs[bt_name] = fs
            b_o = model.addVar(lb=0.0, ub=1.0, name=f"bel_b[{k.id},{other}]")
            bo_name = f"bel_b_def[{k.id},{other}]"
            model.addCons(b_o * ev_v == _e(bp[other]) * _e(q), name=bo_name)
            if factor_specs is not None:
                fs = factor("belief_posterior", "eq", output=b_o)
                add_product(fs, b_o, ev_v, 1.0,
                            roles=("output", "evidence"))
                add_product(fs, bp[other], q, -1.0)
                factor_specs[bo_name] = fs
            model.addCons(b_t + b_o == 1.0, name=f"bel_b_sum[{k.id}]")
            b[(k.id, t)], b[(k.id, other)] = b_t, b_o
            st["n_var"] += 2
            st["n_rows"] += 3
            st["n_bilinear"] += 2

    return w, b, st


def numeric_belief_tree(tree, ac: Dict[Tuple[int, str], float], *,
                        sigma: float, Hb: int, root_belief: Dict[str, float]):
    """The same recursion in plain floats -- the reference the encoding is
    checked against. With `ac` taken from the nominal it must reproduce
    `node.p` and `node.belief` exactly."""
    var = sigma * sigma
    w = {tree.root: 1.0}
    b = {(tree.root, t): float(root_belief[t]) for t in THETAS}
    kids_of = {}
    for n in tree.traverse():
        if n.parent is not None:
            kids_of.setdefault(n.parent, []).append(n)

    for p in tree.traverse():
        kids = kids_of.get(p.id)
        if not kids:
            continue
        d = float(ac[(p.id, "agg")]) - float(ac[(p.id, "cau")])
        q = math.exp(-0.5 * d * d / var)
        for k in kids:
            t = k.tag
            other = "agg" if t == "cau" else "cau"
            ev = b[(p.id, t)] + b[(p.id, other)] * q
            w[k.id] = (w[p.id] * b[(p.id, t)]
                       if p.depth < Hb else w[p.id])
            b[(k.id, t)] = b[(p.id, t)] / ev
            b[(k.id, other)] = b[(p.id, other)] * q / ev
    return w, b
