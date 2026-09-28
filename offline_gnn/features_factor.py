"""variable↔factor graph features for the canonical dual MINLP.

Linear SCIP rows remain ordinary factor nodes.  Every authored nonlinear row
is replaced by a typed factor from ``MINLPBuilder.factor_specs``.
Edges carry both the local Jacobian and symbolic operation/operand roles, so an
edge survives even when its derivative happens to be zero at the root LP.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np


from .feature_utils import (
    _no_scip_output, _VAR_PATTERNS, _CONS_PATTERNS, _VAR_KEYS, _CONS_KEYS,
    _D_LONG, _WL, _V_MAX, _parse_node_id, _classify as _base_classify,
    _safe_lb, _safe_ub,
    _var_key,
)


# Base SCIP bipartite features. These used to be split between
# features_multi.py and features_fast.py; Scheme B is now self-contained here.
_CONS_PATTERNS_MULTI = dict(_CONS_PATTERNS)
_CONS_PATTERNS_MULTI["safety"] = re.compile(
    r"^region_sum(?:_bg\d+)?\[|^front(?:_bg\d+)?\[|^back(?:_bg\d+)?\["
    r"|^left(?:_bg\d+)?\[|^right(?:_bg\d+)?\[|^inner\[|^outer\[")
_SAFETY_IDX_MULTI = _CONS_KEYS.index("safety")
_BG_IDX_RE = re.compile(r"_bg(\d+)")
D_C_BASE = 22


def _bg_index(name: str):
    match = _BG_IDX_RE.search(name)
    return int(match.group(1)) if match else None


def _base_cons_feature_names() -> list[str]:
    return ([f"sem_{key}" for key in _CONS_KEYS] +
            ["norm_bias", "is_eq", "is_tight", "slack", "dual",
             "cos_sim", "density", "t_norm", "p_node",
             "is_focus_opponent", "has_opp_info",
             "opp_x_rel", "opp_y_rel", "opp_vx_rel"])


def _linearize_sqdist(spec, lp_of):
    """Linearize an authored squared-distance row for graph features only."""
    if spec[0] != "sqdist":
        return None
    _, g2, xe, xo, ye, yo = spec
    vals = {name: lp_of(name) for name in (g2, xe, xo, ye)}
    if any(value is None for value in vals.values()):
        return None
    dx, dy = vals[xe] - vals[xo], vals[ye] - yo
    coefs = {g2: 1.0, xe: -2.0 * dx, xo: 2.0 * dx, ye: -2.0 * dy}
    residual = vals[g2] - dx * dx - dy * dy
    rhs = sum(coefs[name] * vals[name] for name in coefs) - residual
    return coefs, rhs


def _extract_base_features(model, builder) -> dict:
    """Sparse O(nnz) SCIP variable/linear-row feature extraction."""
    variables = model.getVars(transformed=False)
    n_vars = len(variables)
    var_index = {var.name: i for i, var in enumerate(variables)}
    lp = np.asarray([var.getLPSol() for var in variables], dtype=np.float64)
    obj = np.asarray([var.getObj() for var in variables], dtype=np.float64)
    obj_norm = float(np.linalg.norm(obj)) + 1e-8
    obj_max = float(np.abs(obj).max()) + 1e-8 if n_vars else 1e-8
    lp_by_name = {var.name: float(lp[i]) for i, var in enumerate(variables)}
    nl_rows = getattr(builder, "nl_rows", None) or {}
    constraints, row_dicts, linearized_rhs = [], [], {}
    with _no_scip_output():
        for cons in model.getConss():
            try:
                row = model.getValsLinear(cons)
                if row:
                    constraints.append(cons)
                    row_dicts.append(row)
            except Exception:
                spec = nl_rows.get(cons.name)
                if spec is None:
                    continue
                lin = _linearize_sqdist(
                    spec, lambda name: lp_by_name.get(name, lp_by_name.get("t_" + name)))
                if lin is not None:
                    constraints.append(cons)
                    row_dicts.append(lin[0])
                    linearized_rhs[cons.name] = lin[1]

    node_lookup, focus_lookup, bg_lookups = {}, {}, {}
    H = max(int(getattr(builder, "H", 1)), 1)
    for nid, node in builder.tree.nodes.items():
        node_lookup[nid] = (node.depth / H, node.p)
        if node.x_o is not None and node.x_e is not None:
            focus_lookup[nid] = (
                (node.x_o[0] - node.x_e[0]) / _D_LONG,
                (node.x_o[1] - node.x_e[1]) / _WL,
                (node.x_o[2] - node.x_e[2]) / _V_MAX)
        if node.x_e is not None:
            for k, path in enumerate(getattr(builder, "bg_paths", [])):
                if path:
                    state = path[min(node.depth, len(path) - 1)]
                    bg_lookups.setdefault(k, {})[nid] = (
                        (float(state[0]) - node.x_e[0]) / _D_LONG,
                        (float(state[1]) - node.x_e[1]) / _WL,
                        (float(state[2]) - node.x_e[2]) / _V_MAX)

    def tree_features(name):
        nid = _parse_node_id(name)
        return node_lookup.get(nid, (0.0, 0.0)) if nid is not None else (0.0, 0.0)

    variable_features = []
    for i, var in enumerate(variables):
        vtype = var.vtype()
        lower, upper, solution = _safe_lb(var), _safe_ub(var), float(lp[i])
        fraction = solution - math.floor(solution)
        try:
            col = var.getCol()
            reduced_cost = (model.getVarRedcost(var) / obj_max
                            if col is not None and col.isInLP() else 0.0)
            if not math.isfinite(reduced_cost):
                reduced_cost = 0.0
        except Exception:
            reduced_cost = 0.0
        t_norm, probability = tree_features(var.name)
        variable_features.append([
            float(vtype == "BINARY"), float(vtype == "INTEGER"),
            float(vtype == "CONTINUOUS"),
            *_base_classify(var.name, _VAR_PATTERNS, _VAR_KEYS),
            float(obj[i]) / obj_max, float(lower > -1e20), float(upper < 1e20),
            solution, float(abs(solution - lower) < 1e-6),
            float(abs(solution - upper) < 1e-6), min(fraction, 1.0 - fraction),
            reduced_cost, t_norm, probability])
    V = np.clip(np.asarray(variable_features, dtype=np.float64),
                -1e10, 1e10).astype(np.float32)

    constraint_features, edge_rows, edge_cols, edge_values = [], [], [], []
    for cons, row in zip(constraints, row_dicts):
        semantic = _base_classify(cons.name, _CONS_PATTERNS_MULTI, _CONS_KEYS)
        if cons.name in linearized_rhs:
            rhs = lhs = linearized_rhs[cons.name]
        else:
            rhs, lhs = model.getRhs(cons), model.getLhs(cons)
        is_eq = float(abs(rhs) < 1e20 and abs(lhs) < 1e20 and abs(rhs - lhs) < 1e-8)
        bias = rhs if abs(rhs) < 1e20 else (lhs if abs(lhs) < 1e20 else 0.0)
        pairs = []
        for var, coef in row.items():
            name = var if isinstance(var, str) else var.name
            key = name if name in var_index else _var_key(name)
            if key in var_index:
                pairs.append((var_index[key], float(coef)))
        indices = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
        coefs = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        if len(coefs):
            l1 = float(np.abs(coefs).sum()) + 1e-8
            l2 = float(np.linalg.norm(coefs)) + 1e-8
            activity = float(coefs @ lp[indices])
            cosine = float((coefs @ obj[indices]) / (l2 * obj_norm))
        else:
            l1 = l2 = 1e-8
            activity = cosine = 0.0
        slack = min(abs(rhs - activity) if rhs < 1e20 else float("inf"),
                    abs(activity - lhs) if lhs > -1e20 else float("inf"))
        try:
            with _no_scip_output():
                dual = model.getDualsolLinear(cons) / obj_max
        except Exception:
            dual = 0.0
        nid = _parse_node_id(cons.name)
        t_norm, probability = node_lookup.get(nid, (0.0, 0.0))
        is_safety = semantic[_SAFETY_IDX_MULTI] > 0.5
        bg_index = _bg_index(cons.name) if is_safety else None
        if is_safety and bg_index is None and nid in focus_lookup:
            is_focus, has_opponent = 1.0, 1.0
            relative = focus_lookup[nid]
        elif is_safety and bg_index in bg_lookups and nid in bg_lookups[bg_index]:
            is_focus, has_opponent = 0.0, 1.0
            relative = bg_lookups[bg_index][nid]
        else:
            is_focus = has_opponent = 0.0
            relative = (0.0, 0.0, 0.0)
        constraint_features.append([
            *semantic, bias / l1, is_eq, float(slack < 1e-6), slack / l1,
            dual, cosine, len(row) / max(n_vars, 1), t_norm, probability,
            is_focus, has_opponent, *relative])
        for index, coef in pairs:
            edge_rows.append(len(constraint_features) - 1)
            edge_cols.append(index)
            edge_values.append(coef / l2)

    C = (np.clip(np.asarray(constraint_features, dtype=np.float64), -1e10, 1e10)
         .astype(np.float32) if constraint_features
         else np.empty((0, D_C_BASE), dtype=np.float32))
    return {
        "variable_features": V,
        "constraint_features": C,
        "edge_index": (np.asarray([edge_rows, edge_cols], dtype=np.int64)
                       if edge_rows else np.empty((2, 0), dtype=np.int64)),
        "edge_features": (np.asarray(edge_values, dtype=np.float32).reshape(-1, 1)
                          if edge_values else np.empty((0, 1), dtype=np.float32)),
        "var_name_order": [var.name for var in variables],
        "cons_name_order": [cons.name for cons in constraints],
    }


VAR_FAMILIES = (
    "ego_state", "control", "lane_idx", "lane_ref", "lane_pos", "lane_neg",
    "lane_slack", "focus_gamma", "background_gamma", "legacy_obj", "goal_abs",
    "opponent_position", "opponent_speed", "gate_distance", "gate_binary",
    "opponent_vref", "raw_accel", "clip_min", "opponent_accel", "clip_hi",
    "clip_lo", "safety_slack", "belief_delta", "belief_likelihood",
    "belief_evidence", "belief_weight", "belief_state", "node_cost",
    "weighted_cost", "nonlinear_aux", "other",
)

_VAR_RULES = (
    ("ego_state", r"^x\["), ("control", r"^u\["),
    ("lane_idx", r"^lane_idx\["), ("lane_ref", r"^r_ref\["),
    ("lane_pos", r"^b_pos\["), ("lane_neg", r"^b_neg\["),
    ("lane_slack", r"^slack_lat\["),
    ("background_gamma", r"^gamma_bg\d+_"), ("focus_gamma", r"^gamma_"),
    ("legacy_obj", r"^t_n\["), ("goal_abs", r"^abs_goal\["),
    ("opponent_position", r"^xo\["), ("opponent_speed", r"^vxo\["),
    ("gate_distance", r"^g2\["), ("gate_binary", r"^z_int\["),
    ("opponent_vref", r"^vref\["), ("raw_accel", r"^araw\["),
    ("clip_min", r"^w_min\["), ("opponent_accel", r"^ac\["),
    ("clip_hi", r"^clip_hi\["), ("clip_lo", r"^clip_lo\["),
    ("safety_slack", r"^slack_safe\["), ("belief_delta", r"^bel_d\["),
    ("belief_likelihood", r"^bel_q\["), ("belief_evidence", r"^bel_E\["),
    ("belief_weight", r"^bel_w\["), ("belief_state", r"^bel_b\["),
    ("node_cost", r"^cost_n\["), ("weighted_cost", r"^tw_n\["),
    ("nonlinear_aux", r"^(?:n|lead_\w+|sstar|head_\w+|qratio(?:_sat)?|afollow|sgap(?:_b)?|sel|pick|a_follow|a_min|min_b|opp_stop|opp_dx|opp_aexp|center_error)\["),
)

CONS_FAMILIES = (
    "initial", "ego_dynamics", "lane_recursion", "lane_dwell", "bound",
    "lane_band", "lane_lock", "safety_region", "safety_separation",
    "gate", "vref_gate", "raw_accel", "clip", "opponent_root",
    "opponent_dynamics", "belief_delta", "belief_likelihood",
    "belief_evidence", "belief_weight", "belief_posterior", "belief_sum",
    "node_cost", "weighted_objective", "goal_abs", "other",
)

_CONS_RULES = (
    ("bound", r"^(?:ahead_|lat_|and_|sgap_|sel_|argmin\[|pick_|a_follow_def|qratio_|afollow_|head_|sstar_|min_|stop_|opp_aexp_|road_)") ,
    ("lane_band", r"^center_error_def"),
    ("initial", r"^x0\[|^lane_idx_root$"), ("ego_dynamics", r"^dyn\["),
    ("lane_recursion", r"^lane_rec\[|^one_dir\["), ("lane_dwell", r"^dwell_"),
    ("bound", r"^u_(?:min|max)\[|^v[xy]_(?:min|max)\["),
    ("lane_band", r"^lane_(?:upper|lower)\[|^r_ref_affine\["),
    ("lane_lock", r"^lane_lock_"), ("safety_region", r"^region_sum"),
    ("safety_separation", r"^(?:front|back|left|right)(?:_bg\d+)?\["),
    ("gate", r"^gate_(?:def|in|out)\["), ("vref_gate", r"^vref_"),
    ("raw_accel", r"^araw_def\["), ("clip", r"^clip_"),
    ("opponent_root", r"^opp_root_"), ("opponent_dynamics", r"^opp_dyn_"),
    ("belief_delta", r"^bel_d_def\["),
    ("belief_likelihood", r"^bel_q_def\["),
    ("belief_evidence", r"^bel_E_def\["),
    ("belief_weight", r"^bel_w_def\["),
    ("belief_posterior", r"^bel_b_def\["),
    ("belief_sum", r"^bel_b_sum\["), ("node_cost", r"^cost_n_def\["),
    ("weighted_objective", r"^obj_w\["), ("goal_abs", r"^abs_goal_"),
)

FACTOR_KINDS = (
    "linear", "squared_distance", "belief_likelihood", "belief_evidence",
    "belief_weight", "belief_posterior", "opponent_belief_dynamics",
    "node_cost", "weighted_objective", "other",
)

EDGE_FEATURES = (
    "jacobian_normalized", "op_linear", "op_square", "op_bilinear",
    "op_exponential", "is_output", "operand_slot",
)

_NID = re.compile(r"\[(\d+)")


def _original(name: str) -> str:
    return name[2:] if name.startswith("t_") else name


def _one_hot(value: str, vocabulary: tuple[str, ...]) -> list[float]:
    return [1.0 if value == item else 0.0 for item in vocabulary]


def _classify(name: str, rules, default="other") -> str:
    for family, pattern in rules:
        if re.search(pattern, name):
            return family
    return default


def _term_value_grad(term: dict, values: dict[str, float]):
    """Return term value and symbolic gradients keyed by variable name."""
    op, names = term["op"], term["vars"]
    coef = float(term.get("coef", 1.0))
    x = [float(values[n]) for n in names]
    if op == "linear":
        return coef * x[0], {names[0]: coef}
    if op == "square":
        d = x[0] - float(term.get("offset", 0.0))
        return coef * d * d, {names[0]: 2.0 * coef * d}
    if op == "square_diff":
        d = x[0] - x[1]
        return coef * d * d, {names[0]: 2.0 * coef * d,
                              names[1]: -2.0 * coef * d}
    if op == "bilinear":
        return coef * x[0] * x[1], {names[0]: coef * x[1],
                                    names[1]: coef * x[0]}
    if op == "exp_neg_square":
        scale = float(term["scale"])
        ev = math.exp(max(-700.0, min(700.0, scale * x[0] * x[0])))
        return coef * ev, {names[0]: coef * ev * 2.0 * scale * x[0]}
    raise ValueError(f"unsupported factor operation {op!r}")


def extract_factor_features(model, builder) -> dict:
    """Build the complete graph after the root LP feature pass."""
    base = _extract_base_features(model, builder)
    specs = getattr(builder, "factor_specs", None) or {}
    all_vars = model.getVars(transformed=False)
    # Original names can legitimately start with t_ (for example t_n[0]).
    var_names = [v.name for v in all_vars]
    var_index = {name: i for i, name in enumerate(var_names)}
    lp = {name: float(all_vars[i].getLPSol()) for i, name in enumerate(var_names)}

    # Extend variable nodes with canonical semantics and tree/global context.
    H = max(int(getattr(builder, "H", 1)), 1)
    tree = builder.tree
    split_depths = [n.depth for n in tree.traverse() if len(tree.children(n.id)) > 1]
    hb = max(split_depths) + 1 if split_depths else 0
    root_b = tree.nodes[tree.root].belief
    p_agg = float(root_b.get("agg", 0.5))
    entropy = -sum(float(q) * math.log(max(float(q), 1e-12)) for q in root_b.values())
    v_extra = []
    unknown_vars = []
    for name in var_names:
        fam = _classify(name, _VAR_RULES)
        if fam == "other":
            unknown_vars.append(name)
        match = _NID.search(name)
        nid = int(match.group(1)) if match and int(match.group(1)) in tree.nodes else None
        node = tree.nodes[nid] if nid is not None else None
        v_extra.append([
            *_one_hot(fam, VAR_FAMILIES),
            1.0 if node is not None else 0.0,
            1.0 if node is not None and nid == tree.root else 0.0,
            1.0 if node is not None and not tree.children(nid) else 0.0,
            1.0 if node is not None and node.depth < hb else 0.0,
            1.0 if node is not None and node.tag == "cau" else 0.0,
            1.0 if node is not None and node.tag == "agg" else 0.0,
            p_agg, entropy,
        ])
    V = np.concatenate([base["variable_features"], np.asarray(v_extra, np.float32)], axis=1)

    # Retain ordinary SCIP rows, removing nonlinear rows that typed factors replace.
    old_names = list(base["cons_name_order"])
    keep_old = np.array([_original(n) not in specs for n in old_names], dtype=bool)
    old_to_new = {}
    kept_names = []
    for old_i, (name, keep) in enumerate(zip(old_names, keep_old)):
        if keep:
            old_to_new[old_i] = len(kept_names)
            kept_names.append(_original(name))
    C0 = base["constraint_features"][keep_old]
    cons_extra = []
    unknown_cons = []
    for name in kept_names:
        fam = _classify(name, _CONS_RULES)
        if fam == "other":
            unknown_cons.append(name)
        cons_extra.append([*_one_hot(fam, CONS_FAMILIES),
                           *_one_hot("linear", FACTOR_KINDS), 0.0, 0.0])

    edge_rows, edge_cols, edge_vals = [], [], []
    for edge_no in range(base["edge_index"].shape[1]):
        old_c = int(base["edge_index"][0, edge_no])
        if old_c not in old_to_new:
            continue
        edge_rows.append(old_to_new[old_c])
        edge_cols.append(int(base["edge_index"][1, edge_no]))
        jac = float(base["edge_features"][edge_no, 0])
        edge_vals.append([jac, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Add one typed factor per authored nonlinear relation.
    factor_names = []
    for cname, spec in specs.items():
        row = len(kept_names) + len(factor_names)
        factor_names.append(cname)
        fam = _classify(cname, _CONS_RULES)
        if fam == "other":
            unknown_cons.append(cname)
        residual = float(spec.get("constant", 0.0))
        grads = defaultdict(float)
        edge_meta = defaultdict(lambda: {"ops": set(), "output": False,
                                         "slots": []})
        for term in spec["terms"]:
            value, term_grads = _term_value_grad(term, lp)
            residual += value
            for slot, name in enumerate(term["vars"]):
                grads[name] += term_grads[name]
                edge_meta[name]["ops"].add(term["op"])
                edge_meta[name]["slots"].append(slot)
        for name in spec.get("outputs", []):
            edge_meta[name]["output"] = True
        scale = math.sqrt(sum(g * g for g in grads.values())) + 1e-8
        kind = spec.get("kind", "other")
        cons_extra.append([*_one_hot(fam, CONS_FAMILIES),
                           *_one_hot(kind if kind in FACTOR_KINDS else "other",
                                     FACTOR_KINDS),
                           1.0, residual / scale])
        for name, meta in edge_meta.items():
            if name not in var_index:
                raise KeyError(f"factor {cname} references absent variable {name}")
            ops = meta["ops"]
            slot = float(sum(meta["slots"]) / max(len(meta["slots"]), 1))
            edge_rows.append(row)
            edge_cols.append(var_index[name])
            edge_vals.append([
                grads[name] / scale,
                1.0 if "linear" in ops else 0.0,
                1.0 if ("square" in ops or "square_diff" in ops) else 0.0,
                1.0 if "bilinear" in ops else 0.0,
                1.0 if "exp_neg_square" in ops else 0.0,
                1.0 if meta["output"] else 0.0,
                slot,
            ])

    C_extra = np.asarray(cons_extra, dtype=np.float32)
    # Typed factor features are appended to the legacy 22 columns. New factors
    # have no meaningful legacy row statistics, except equality and residual.
    C_factor_legacy = np.zeros((len(factor_names), C0.shape[1]), dtype=np.float32)
    legacy_names = _base_cons_feature_names()
    if factor_names:
        eq_i = legacy_names.index("is_eq")
        slack_i = legacy_names.index("slack")
        for i, cname in enumerate(factor_names):
            C_factor_legacy[i, eq_i] = 1.0 if specs[cname]["sense"] == "eq" else 0.0
            C_factor_legacy[i, slack_i] = abs(C_extra[len(kept_names) + i, -1])
    C_legacy = np.concatenate([C0, C_factor_legacy], axis=0)
    C = np.concatenate([C_legacy, C_extra], axis=1)

    E = np.asarray(edge_vals, dtype=np.float32)
    edge_index = np.asarray([edge_rows, edge_cols], dtype=np.int64)
    degrees = np.bincount(edge_index[1], minlength=len(var_names)) if edge_cols else np.zeros(len(var_names), int)
    is_binary = base["variable_features"][:, 0] > 0.5
    families = [_classify(name, _VAR_RULES) for name in var_names]
    eligible = np.array([
        is_binary[i] and families[i] in {
            "lane_pos", "lane_neg", "focus_gamma", "background_gamma"
        } and not (families[i] in {"focus_gamma", "background_gamma"} and
                   (_NID.search(name) and int(_NID.search(name).group(1)) == tree.root))
        for i, name in enumerate(var_names)
    ], dtype=bool)
    nonlinear_names = {
        _original(c.name) for c in model.getConss()
        if c.getConshdlrName() == "nonlinear"
    }
    isolated_eligible = eligible & (degrees == 0)
    candidate = eligible & (degrees > 0)
    return {
        "variable_features": V,
        "constraint_features": C,
        "edge_index": edge_index,
        "edge_features": E,
        "var_name_order": var_names,
        "cons_name_order": kept_names + factor_names,
        "factor_names": factor_names,
        "factor_specs": specs,
        "unknown_variable_names": unknown_vars,
        "unknown_constraint_names": unknown_cons,
        "candidate_mask": candidate,
        "candidate_isolated": [var_names[i] for i in np.where(isolated_eligible)[0]],
        "nonlinear_constraint_names": sorted(nonlinear_names),
        "missing_nonlinear_factors": sorted(nonlinear_names - set(specs)),
        "feature_names": {
            "variable_extra": [f"family_{x}" for x in VAR_FAMILIES] +
                ["has_tree_node", "is_root", "is_leaf", "pre_Hb",
                 "tag_cau", "tag_agg", "root_belief_agg", "root_belief_entropy"],
            "constraint_extra": [f"family_{x}" for x in CONS_FAMILIES] +
                [f"factor_{x}" for x in FACTOR_KINDS] +
                ["is_nonlinear_factor", "factor_residual"],
            "edge": list(EDGE_FEATURES),
        },
    }
