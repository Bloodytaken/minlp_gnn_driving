"""Two-type scenario tree for the dual MINLP.

Branch once per cautious/aggressive type, without process-noise samples.
After the branching horizon, freeze weights and propagate the belief mean."""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


from opponent_model.traffic_model import following_model
from opponent_model.traffic_follow import step_stop_aware
from tree import Node, Tree
from opponent_model.opponent import OpponentModel
from problem_setup import make_roll_dynamics, minlp_to_tree_state

THETA_SIGN = {"cau": -1.0, "agg": +1.0}
THETAS = ("cau", "agg")
SIGMA_OBS = 1.0


def bayes_2(prior: Dict[str, float], lik: Dict[str, float],
            floor: float = 0.0) -> Dict[str, float]:
    """Two-hypothesis Bayes update. `floor` clamps the posterior away from 0.

    Deliberately NOT `belief.bayes_update`: that one carries an
    `evidence <= 1e-12 -> keep the prior` fallback, which is unrepresentable
    inside a MINLP (an indicator on a quantity spanning 18 orders of magnitude,
    against SCIP's 1e-6 feasibility tolerance). Here the likelihoods are scaled
    so the larger is exactly 1 -- an exactly invariant rescaling, since both the
    posterior and the weights are ratios -- which keeps the evidence O(1) and
    makes the fallback unnecessary.
    """
    m = max(lik[t] for t in THETAS)
    if m <= 0.0:
        return dict(prior)
    scaled = {t: lik[t] / m for t in THETAS}
    unnorm = {t: prior[t] * scaled[t] for t in THETAS}
    ev = sum(unnorm.values())
    if ev <= 0.0:
        return dict(prior)
    post = {t: unnorm[t] / ev for t in THETAS}
    if floor > 0.0:
        post = {t: min(max(post[t], floor), 1.0 - floor) for t in THETAS}
        s = sum(post.values())
        post = {t: post[t] / s for t in THETAS}
    return post


def build_dual_tree(H: int, Hb: int, dt: float, x_e0, x_o0, *,
                    ego_controls=None, prior: Optional[Dict[str, float]] = None,
                    opp: Optional[OpponentModel] = None,
                    belief_floor: float = 0.0):
    """Build a two-way belief tree; return (tree, opponent, type mean functions)."""
    opp = following_model(opp)
    if prior is None:
        prior = {"cau": 0.5, "agg": 0.5}
    if ego_controls is None:
        ego_controls = [(0.0, 0.0)] * H
    f_e, f_o = make_roll_dynamics(dt, center=(0.0, 0.0), force_straight=True)
    # dual_controller.py uses a dedicated observation model with sigma_obs=1;
    # it is not the opponent process-noise parameter.
    sigma = SIGMA_OBS
    var = sigma * sigma

    mu_fns = {t: (lambda xo, xe, s=THETA_SIGN[t]: opp.mean_accel_scalar(xo, xe, s))
              for t in THETAS}

    def ue(depth):
        i = min(depth, len(ego_controls) - 1)
        c = ego_controls[i] if i < len(ego_controls) else (0.0, 0.0)
        return (float(c[0]), float(c[1])) if c else (0.0, 0.0)

    tree = Tree(root=0)
    root = Node(id=0, depth=0, parent=None, p=1.0, opp_u=None, tag=None,
                belief=dict(prior),
                x_e=tuple(minlp_to_tree_state(x_e0)),
                x_o=tuple(minlp_to_tree_state(x_o0)))
    tree.add(root)
    nid = 1
    frontier = [0]

    while frontier:
        pid = frontier.pop(0)
        p = tree.nodes[pid]
        if p.depth >= H:
            continue
        x_e_next = f_e(p.x_e, ue(p.depth))
        ac = {t: float(opp.mean_accel_scalar(p.x_o, p.x_e, THETA_SIGN[t], depth=p.depth)) for t in THETAS}

        if p.depth < Hb:
            # branching: one child per driver type, weight = parent * b(theta)
            for t in THETAS:
                lik = {th: math.exp(-0.5 * (ac[t] - ac[th]) ** 2 / var)
                       for th in THETAS}
                child = Node(
                    id=nid, depth=p.depth + 1, parent=pid,
                    p=p.p * p.belief[t],
                    opp_u=ac[t], tag=t,
                    belief=bayes_2(p.belief, lik, floor=belief_floor),
                    x_e=x_e_next, x_o=step_stop_aware(p.x_o, ac[t], dt),
                )
                tree.add(child)
                frontier.append(nid)
                nid += 1
        else:
            # Reference propagation: keep the inherited branch tag and weight,
            # update belief from that tag's acceleration, but propagate the
            # opponent state with the current belief-weighted mean acceleration.
            t = p.tag
            lik = {th: math.exp(-0.5 * (ac[t] - ac[th]) ** 2 / var)
                   for th in THETAS}
            a_exp = sum(float(p.belief[th]) * ac[th] for th in THETAS)
            child = Node(
                id=nid, depth=p.depth + 1, parent=pid, p=p.p,
                opp_u=a_exp, tag=t,
                belief=bayes_2(p.belief, lik, floor=belief_floor),
                x_e=x_e_next, x_o=step_stop_aware(p.x_o, a_exp, dt),
            )
            tree.add(child)
            frontier.append(nid)
            nid += 1

    return tree, opp, mu_fns
