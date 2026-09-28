"""Straight-road dual MINLP with endogenous opponent reactions and beliefs."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyscipopt import Model, quicksum
from tree import Tree, Node
from opponent_model.opponent import OpponentModel


from opponent_model.opponent_law import (LawParams, THETA_SIGN, encode_clip, encode_gate,
                          encode_vref)
from belief_law import encode_belief_tree
from minlp_tree import SIGMA_OBS
from opponent_model.traffic_model import following_model
from opponent_model.follow_law import Candidate, FollowParams, encode_follow, encode_min
from stop_law import encode_stop_step
from road_boundary import add_road_bounds
from offline_gnn.factor_schema import (add_linear, add_product, add_square,
                           add_square_diff, factor)

TreeState = Tuple[float, float, float, float]  # (x, y, vx, vy)


class _BaseFormulation:
    """Base dynamics, lane decisions and safety constraints."""

    def __init__(self, H: int, dt: float, weights: dict, geom: dict, bigM: float = 1e3):
        self.H = H
        self.dt = dt
        self.w = weights
        self.geom = geom
        self.M = bigM

        dt2 = dt * dt
        self.A = np.array(
            [
                [1, dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dt],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.B = np.array(
            [
                [0.5 * dt2, 0.0],
                [dt, 0.0],
                [0.0, 0.5 * dt2],
                [0.0, dt],
            ],
            dtype=float,
        )
        self.center = np.asarray(self.geom["center"], dtype=float)
        self.normals = self._prepare_normals()

    def _prepare_normals(self):
        """Outward unit normals from geom['beta'] or geom['ref_path']."""
        if "beta" in self.geom:
            seq = self.geom["beta"]
            if len(seq) < self.H + 1:
                raise ValueError("beta list must have length ≥ H+1")
            normals = [
                np.array([math.cos(th), math.sin(th)], dtype=float) for th in seq
            ]
        elif "ref_path" in self.geom:
            pts = self.geom["ref_path"]
            if len(pts) < self.H + 1:
                raise ValueError("ref_path list must have length ≥ H+1")
            normals = []
            for p in pts:
                vec = np.asarray(p, dtype=float) - self.center
                norm = np.linalg.norm(vec)
                if norm < 1e-6:
                    normals.append(np.array([1.0, 0.0]))
                else:
                    normals.append(vec / norm)
        else:
            raise ValueError("geom must provide 'beta' or 'ref_path' for normals.")
        return normals[: self.H + 1]

    @staticmethod
    def _tangent_from_normal(n_hat: np.ndarray) -> np.ndarray:
        return np.array([-n_hat[1], n_hat[0]])

    @staticmethod
    def _quad_cost(vars_vec, ref_vec, weights):
        # Python accumulation keeps a single SCIP expression tree.
        acc = 0.0
        for i, w in enumerate(weights):
            if w == 0:
                continue
            diff = vars_vec[i] - float(ref_vec[i])
            acc = acc + float(w) * diff * diff
        return acc

    def _ref_state(self, depth: int):
        seq = self.w.get("x_ref")
        if callable(seq):
            return seq(depth)
        if isinstance(seq, (list, tuple)):
            if len(seq) <= depth:
                raise ValueError("x_ref list shorter than horizon.")
            return seq[depth]
        return np.zeros(4)

    def _ref_control(self, depth: int):
        seq = self.w.get("u_ref")
        if callable(seq):
            return seq(depth)
        if isinstance(seq, (list, tuple)):
            if len(seq) <= depth:
                raise ValueError("u_ref list shorter than horizon.")
            return seq[depth]
        return np.zeros(2)

    def _opponent_pose(self, node: Node):
        if node.x_o is None:
            raise ValueError("Tree node is missing opponent state for MINLP.")
        # Tree order: (x, y, vx, vy); optimizer order: (x, vx, y, vy).
        return np.array([node.x_o[0], node.x_o[1]], dtype=float)

    def build(self, x0, tree: Tree, belief, opp: OpponentModel):
        del belief, opp
        m = Model("straight_dual_minlp")
        m.setIntParam("display/verblevel", 0)

        inf = m.infinity()

        nx, nu = 4, 2
        nodes = list(tree.traverse())
        depth_weights = defaultdict(float)
        for node in nodes:
            depth_weights[node.depth] += node.p

        X = {
            node.id: [
                m.addVar(lb=-inf, ub=inf, name=f"x[{node.id},{i}]") for i in range(nx)
            ]
            for node in nodes
        }
        # One control per node enforces non-anticipativity.
        U = {
            node.id: [
                m.addVar(lb=-inf, ub=inf, name=f"u[{node.id},{j}]") for j in range(nu)
            ]
            for node in nodes
            if node.depth < self.H
        }
        lane_idx = {
            node.id: m.addVar(vtype="I", lb=0, ub=self.geom["M"] - 1, name=f"lane_idx[{node.id}]")
            for node in nodes
        }
        r_ref = {
            node.id: m.addVar(
                lb=self.geom["R0"] + 0.5 * self.geom["wl"],
                ub=self.geom["R0"] + (self.geom["M"] - 0.5) * self.geom["wl"],
                name=f"r_ref[{node.id}]"
            )
            for node in nodes
        }
        # Lane-transition binaries are indexed by the parent node.
        b_pos = {
            node.id: m.addVar(vtype="B", name=f"b_pos[{node.id}]")
            for node in nodes
            if node.depth < self.H
        }
        b_neg = {
            node.id: m.addVar(vtype="B", name=f"b_neg[{node.id}]")
            for node in nodes
            if node.depth < self.H
        }
        # Block a new lane command while the previous transition is in progress.
        if self.geom.get("lock_lane_changes", False):
            for nid in b_pos:
                m.addCons(b_pos[nid] == 0, name=f"lane_lock_pos[{nid}]")
                m.addCons(b_neg[nid] == 0, name=f"lane_lock_neg[{nid}]")
        slack_lat = {
            node.id: m.addVar(lb=0.0, name=f"slack_lat[{node.id}]") for node in nodes
        }

        for i in range(nx):
            m.addCons(X[tree.root][i] == float(x0[i]), name=f"x0[{i}]")
        m.addCons(
            lane_idx[tree.root] == int(self.geom.get("lane0", 0)),
            name="lane_idx_root",
        )
        wl = self.geom["wl"]
        R0 = self.geom["R0"]
        for node in nodes:
            m.addCons(r_ref[node.id] == R0 + wl * (lane_idx[node.id] + 0.5), name=f"r_ref_affine[{node.id}]")

        for node in nodes:
            if node.id == tree.root:
                continue
            parent = tree.nodes[node.parent]
            m.addCons(
                lane_idx[node.id] == lane_idx[parent.id] + b_pos[parent.id] - b_neg[parent.id],
                name=f"lane_rec[{node.id}]",
            )
            m.addCons(b_pos[parent.id] + b_neg[parent.id] <= 1, name=f"one_dir[{parent.id}]")

        # At most one lane change on each root-to-leaf path.
        leaf_nodes = [n for n in nodes if n.depth == self.H or len(tree.children(n.id)) == 0]
        for leaf in leaf_nodes:
            path_nodes = []
            cur = leaf
            while cur is not None and cur.id != tree.root:
                parent = tree.nodes.get(cur.parent)
                if parent is not None and parent.id in b_pos:
                    path_nodes.append(parent.id)
                cur = parent
            if path_nodes:
                m.addCons(
                    quicksum(b_pos[nid] + b_neg[nid] for nid in path_nodes) <= 1,
                    name=f"dwell_one_change_path_{leaf.id}"
                )

        for node in nodes:
            if node.id == tree.root:
                continue
            parent = tree.nodes[node.parent]
            for i in range(nx):
                dyn_expr = quicksum(self.A[i, j] * X[parent.id][j] for j in range(nx))
                if parent.id in U:
                    dyn_expr += quicksum(self.B[i, j] * U[parent.id][j] for j in range(nu))
                m.addCons(X[node.id][i] == dyn_expr, name=f"dyn[{node.id},{i}]")

        u_min = np.array(self.geom.get("u_min", [-inf] * nu), dtype=float)
        u_max = np.array(self.geom.get("u_max", [inf] * nu), dtype=float)
        v_min = np.array(self.geom.get("v_min", [-inf, -inf]), dtype=float)
        v_max = np.array(self.geom.get("v_max", [inf, inf]), dtype=float)
        for nid, uvars in U.items():
            for j in range(nu):
                if np.isfinite(u_min[j]):
                    m.addCons(uvars[j] >= u_min[j], name=f"u_min[{nid},{j}]")
                if np.isfinite(u_max[j]):
                    m.addCons(uvars[j] <= u_max[j], name=f"u_max[{nid},{j}]")
        for node in nodes:
            if np.isfinite(v_min[0]):
                m.addCons(X[node.id][1] >= v_min[0], name=f"vx_min[{node.id}]")
            if np.isfinite(v_max[0]):
                m.addCons(X[node.id][1] <= v_max[0], name=f"vx_max[{node.id}]")
            if np.isfinite(v_min[1]):
                m.addCons(X[node.id][3] >= v_min[1], name=f"vy_min[{node.id}]")
            if np.isfinite(v_max[1]):
                m.addCons(X[node.id][3] <= v_max[1], name=f"vy_max[{node.id}]")

        vehicle_width = float(self.geom.get("vehicle_width", 2.2))
        lane_margin = max(0.0, 0.5 * wl - 0.5 * vehicle_width)

        for node in nodes:
            if node.id == tree.root:
                continue

            py = X[node.id][2]

            if self.geom.get("is_straight_road", False):
                lane_center_y = self.center[1] + r_ref[node.id]
                e_expr = py - lane_center_y
            else:
                depth = node.depth
                n_hat = self.normals[min(depth, self.H)]
                px = X[node.id][0]
                e_expr = (
                    n_hat[0] * (px - self.center[0])
                    + n_hat[1] * (py - self.center[1])
                    - r_ref[node.id]
                )

            m.addCons(
                e_expr <= lane_margin + slack_lat[node.id],
                name=f"lane_upper[{node.id}]",
            )
            m.addCons(
                e_expr >= -lane_margin - slack_lat[node.id],
                name=f"lane_lower[{node.id}]",
            )

        d_tau = self.geom["d_tau"]
        d_n = self.geom["d_n"]
        Gamma = {}

        for node in nodes:
            p_o = self._opponent_pose(node)

            px = X[node.id][0]
            py = X[node.id][2]

            if self.geom.get("is_straight_road", False):
                r_tau = px - p_o[0]
                r_n = py - p_o[1]
            else:
                vec = p_o - self.center
                norm = np.linalg.norm(vec)
                if norm < 1e-6:
                    n_hat_o = np.array([1.0, 0.0])
                else:
                    n_hat_o = vec / norm
                tau_hat_o = self._tangent_from_normal(n_hat_o)

                r_tau = tau_hat_o[0] * (px - p_o[0]) + tau_hat_o[1] * (py - p_o[1])
                r_n = n_hat_o[0] * (px - p_o[0]) + n_hat_o[1] * (py - p_o[1])

            Gamma[(node.id, "f")] = m.addVar(vtype="B", name=f"gamma_f[{node.id}]")
            Gamma[(node.id, "b")] = m.addVar(vtype="B", name=f"gamma_b[{node.id}]")
            Gamma[(node.id, "l")] = m.addVar(vtype="B", name=f"gamma_l[{node.id}]")
            Gamma[(node.id, "r")] = m.addVar(vtype="B", name=f"gamma_r[{node.id}]")

            m.addCons(
                Gamma[(node.id, "f")]
                + Gamma[(node.id, "b")]
                + Gamma[(node.id, "l")]
                + Gamma[(node.id, "r")] == 1,
                name=f"region_sum[{node.id}]",
            )

            m.addCons(
                r_tau >= d_tau - self.M * (1 - Gamma[(node.id, "f")]),
                name=f"front[{node.id}]",
            )
            m.addCons(
                -r_tau >= d_tau - self.M * (1 - Gamma[(node.id, "b")]),
                name=f"back[{node.id}]",
            )

            m.addCons(
                -r_n >= d_n - self.M * (1 - Gamma[(node.id, "l")]),
                name=f"left[{node.id}]",
            )
            m.addCons(
                r_n >= d_n - self.M * (1 - Gamma[(node.id, "r")]),
                name=f"right[{node.id}]",
            )

        Q = np.asarray(self.w.get("Q", [1.0, 0.0, 1.0, 0.0]), dtype=float)
        Rw = np.asarray(self.w.get("R", [0.1, 0.1]), dtype=float)
        Qf = np.asarray(self.w.get("Qf", Q), dtype=float)
        w_slack = float(self.w.get("slack", 0.0))
        lam_sw = float(self.w.get("lane_switch", 0.0))

        w_goal = float(self.w.get("w_goal", 0.0))
        goal_lane_target = self.w.get("goal_lane", None)

        # One quadratic cost epigraph per node.
        lin_obj_terms = []
        for node in nodes:
            depth = node.depth
            x_ref = self._ref_state(depth)
            Qw = Qf if depth == self.H else Q

            t_n = m.addVar(lb=0.0, name=f"t_n[{node.id}]")
            node_cost = self._quad_cost(X[node.id], x_ref, Qw)
            if node.id in U:
                u_ref = self._ref_control(depth)
                node_cost = node_cost + self._quad_cost(U[node.id], u_ref, Rw)
            m.addCons(t_n >= node.p * node_cost, name=f"obj_node[{node.id}]")
            lin_obj_terms.append(t_n)

            if w_slack > 0:
                lin_obj_terms.append(node.p * w_slack * slack_lat[node.id])
            if lam_sw > 0 and node.id in b_pos:
                lin_obj_terms.append(lam_sw * node.p * (b_pos[node.id] + b_neg[node.id]))
            if w_goal > 0 and goal_lane_target is not None:
                g = float(goal_lane_target)
                ag = m.addVar(lb=0.0, name=f"abs_goal[{node.id}]")
                m.addCons(ag >= lane_idx[node.id] - g, name=f"abs_goal_pos[{node.id}]")
                m.addCons(ag >= g - lane_idx[node.id], name=f"abs_goal_neg[{node.id}]")
                lin_obj_terms.append(node.p * w_goal * ag)

        m.setObjective(quicksum(lin_obj_terms), "minimize")

        self.model = m
        self.tree = tree
        self.X = X
        self.U = U
        self.U_mode = None
        self.lane_idx = lane_idx
        self.b_pos = b_pos
        self.b_neg = b_neg
        self.gamma = Gamma
        return m


class _BackgroundFormulation(_BaseFormulation):
    """Safety regions against given background-vehicle trajectories."""

    def __init__(self, H: int, dt: float, weights: dict, geom: dict,
                 bigM: float = 1e3, bg_paths: Optional[List[List[TreeState]]] = None):
        if type(self) is _BackgroundFormulation:
            raise RuntimeError(
                "incomplete formulation cannot be instantiated; construct "
                "MINLPBuilder instead"
            )
        super().__init__(H=H, dt=dt, weights=weights, geom=geom, bigM=bigM)
        self.bg_paths = list(bg_paths) if bg_paths is not None else []

    def _bg_pose(self, k: int, depth: int) -> np.ndarray:
        path = self.bg_paths[k]
        state = np.asarray(path[min(depth, len(path) - 1)], dtype=float)
        return state[:2]

    def build(self, x0, tree: Tree, belief, opp):
        model = super().build(x0, tree, belief, opp)

        gamma_focus = dict(self.gamma)
        gamma_bg: Dict[Tuple[int, int, str], object] = {}
        d_tau = float(self.geom["d_tau"])
        d_n = float(self.geom["d_n"])

        for k in range(len(self.bg_paths)):
            for node in tree.traverse():
                p_bg = self._bg_pose(k, node.depth)
                px = self.X[node.id][0]
                py = self.X[node.id][2]
                r_tau = px - p_bg[0]
                r_n = py - p_bg[1]

                for key in ("f", "b", "l", "r"):
                    gamma_bg[(node.id, k, key)] = model.addVar(
                        vtype="B", name=f"gamma_bg{k}_{key}[{node.id}]"
                    )
                model.addCons(
                    quicksum(gamma_bg[(node.id, k, key)] for key in ("f", "b", "l", "r")) == 1,
                    name=f"region_sum_bg{k}[{node.id}]",
                )
                model.addCons(
                    r_tau >= d_tau - self.M * (1 - gamma_bg[(node.id, k, "f")]),
                    name=f"front_bg{k}[{node.id}]",
                )
                model.addCons(
                    -r_tau >= d_tau - self.M * (1 - gamma_bg[(node.id, k, "b")]),
                    name=f"back_bg{k}[{node.id}]",
                )
                model.addCons(
                    -r_n >= d_n - self.M * (1 - gamma_bg[(node.id, k, "l")]),
                    name=f"left_bg{k}[{node.id}]",
                )
                model.addCons(
                    r_n >= d_n - self.M * (1 - gamma_bg[(node.id, k, "r")]),
                    name=f"right_bg{k}[{node.id}]",
                )

        self.gamma = gamma_focus
        self.gamma_focus = gamma_focus
        self.gamma_bg = gamma_bg
        return model


@dataclass(frozen=True)
class CouplingConfig:
    """Opponent coupling and bound-tightening options."""

    mode: str = "exact"
    max_depth: Optional[int] = None      # None = full horizon
    tighten_ego_bounds: bool = True

    def __post_init__(self):
        if self.mode != "exact":
            raise ValueError(
                "frozen opponent coupling is not a valid MINLP formulation; "
                "mode must be 'exact'"
            )

    @classmethod
    def from_geom(cls, geom: dict) -> "CouplingConfig":
        raw = (geom or {}).get("opp_coupling")
        if not raw:
            return cls()
        if isinstance(raw, CouplingConfig):
            return raw
        max_depth = raw.get("max_depth")
        cfg = cls(mode=str(raw.get("mode", "exact")),
                  max_depth=(int(max_depth) if max_depth is not None else None),
                  tighten_ego_bounds=bool(raw.get("tighten_ego_bounds", True)))
        return cfg

    @property
    def belief_active(self) -> bool:
        return True


class MINLPBuilder(_BackgroundFormulation):
    """Build the complete dual MINLP over a scenario tree."""

    def __init__(self, H, dt, weights, geom, bigM=1e3, bg_paths=None,
                 coupling: Optional[CouplingConfig] = None):
        super().__init__(H=H, dt=dt, weights=weights, geom=geom, bigM=bigM,
                         bg_paths=bg_paths)
        if coupling is not None and not isinstance(coupling, CouplingConfig):
            raise TypeError("coupling must be an exact CouplingConfig")
        self.coupling = coupling or CouplingConfig.from_geom(geom)
        self.law: Optional[LawParams] = None
        self.xo: Dict[int, object] = {}
        self.vxo: Dict[int, object] = {}
        self.yo_const: Dict[int, float] = {}
        self.z: Dict[int, object] = {}
        self.ac: Dict[Tuple[int, str], object] = {}
        self.coupling_stats: Dict[str, object] = {}
        # Nonlinear rows used for local graph linearization.
        self.nl_rows: Dict[str, tuple] = {}
        # Nonlinear factor metadata for GNN graph extraction.
        self.factor_specs: Dict[str, dict] = {}
        self.w_node: Dict[int, object] = {}
        self.b_node: Dict[Tuple[int, str], object] = {}
        self.belief_stats: Dict[str, object] = {}
        self.slack_safe: Dict[int, object] = {}

    @staticmethod
    def _validate_dual_tree(tree) -> int:
        """Validate two-type branching and return the branching horizon."""
        split_depths = []
        for parent in tree.traverse():
            kids = tree.children(parent.id)
            if not kids:
                continue
            tags = [k.tag for k in kids]
            if len(kids) > 1:
                if len(kids) != 2 or sorted(tags) != ["agg", "cau"]:
                    raise ValueError(
                        "canonical dual MINLP requires exactly two children "
                        "{cau, agg} at every branching node; the old sampled "
                        f"tree has parent {parent.id} tags={tags}")
                split_depths.append(parent.depth)
            elif tags[0] not in THETA_SIGN:
                raise ValueError(f"invalid propagation tag at node {kids[0].id}: {tags[0]!r}")
        if not split_depths:
            return 0
        hb = max(split_depths) + 1
        for parent in tree.traverse():
            kids = tree.children(parent.id)
            expected = 2 if parent.depth < hb and parent.depth < max(
                n.depth for n in tree.traverse()) else 1
            if kids and len(kids) != expected:
                raise ValueError(
                    f"non-reference topology at parent {parent.id}: depth={parent.depth}, "
                    f"expected {expected} children, got {len(kids)}")
        return hb

    def _state_bounds(self, tree, x_o0):
        """Reachable opponent bounds with maximal braking to rest."""
        dt, a_max = self.dt, self.law.a_max
        xb: Dict[int, Tuple[float, float]] = {}
        vb: Dict[int, Tuple[float, float]] = {}
        for node in tree.traverse():
            if node.parent is None:
                # The root equality fixes the state; pad its bounds for numerical tolerance.
                xb[node.id] = self._pad(float(x_o0[0]), float(x_o0[0]))
                vb[node.id] = (max(0., self._pad(float(x_o0[2]), float(x_o0[2]))[0]),
                               self._pad(float(x_o0[2]), float(x_o0[2]))[1])
                continue
            a_lo, a_hi = -a_max, a_max
            pxl, pxh = xb[node.parent]
            pvl, pvh = vb[node.parent]
            vb[node.id] = (max(0.0, pvl + a_lo * dt), self._pad(0., pvh + a_hi * dt)[1])
            min_move = (pvl*dt-.5*a_max*dt*dt if pvl >= a_max*dt
                        else pvl*pvl/(2*a_max))
            xb[node.id] = self._pad(pxl + min_move,
                                    pxh + pvh * dt + 0.5 * a_hi * dt * dt)
        return xb, vb

    @staticmethod
    def _pad(lo: float, hi: float) -> Tuple[float, float]:
        """Pad reachable bounds to avoid floating-point infeasibility."""
        pad_lo = 1e-6 + 1e-9 * abs(lo)
        pad_hi = 1e-6 + 1e-9 * abs(hi)
        return (lo - pad_lo if np.isfinite(lo) else lo,
                hi + pad_hi if np.isfinite(hi) else hi)

    def _ego_speed_bounds(self, x0, depth) -> Tuple[float, float]:
        """Reachable longitudinal speed bounds for tighter big-M constants."""
        dt = self.dt
        vmin = np.asarray(self.geom.get("v_min", [-1e20, -1e20]), dtype=float)
        vmax = np.asarray(self.geom.get("v_max", [1e20, 1e20]), dtype=float)
        umin = np.asarray(self.geom.get("u_min", [-1e20, -1e20]), dtype=float)
        umax = np.asarray(self.geom.get("u_max", [1e20, 1e20]), dtype=float)
        span = depth * dt
        lo = max(float(vmin[0]), float(x0[1]) + float(umin[0]) * span)
        hi = min(float(vmax[0]), float(x0[1]) + float(umax[0]) * span)
        return self._pad(lo, hi)

    def _ego_bounds(self, x0, depth):
        """Reachable position bounds implied by endpoint velocity bounds."""
        dt = self.dt
        vmin = np.asarray(self.geom.get("v_min", [-1e20, -1e20]), dtype=float)
        vmax = np.asarray(self.geom.get("v_max", [1e20, 1e20]), dtype=float)
        span = depth * dt
        x_lo = float(x0[0]) + (vmin[0] * span if np.isfinite(vmin[0]) else -1e20)
        x_hi = float(x0[0]) + (vmax[0] * span if np.isfinite(vmax[0]) else 1e20)
        y_lo = float(x0[2]) + (vmin[1] * span if np.isfinite(vmin[1]) else -1e20)
        y_hi = float(x0[2]) + (vmax[1] * span if np.isfinite(vmax[1]) else 1e20)
        return self._pad(x_lo, x_hi), self._pad(y_lo, y_hi)

    def build(self, x0, tree, belief, opp):
        opp = following_model(opp)
        hb = self._validate_dual_tree(tree)
        model = super().build(x0, tree, belief, opp)

        self.law = LawParams.from_model(opp)
        p = self.law
        dt = self.dt
        cfg = self.coupling
        if cfg.max_depth is not None and int(cfg.max_depth) < self.H - 1:
            raise ValueError("partial opponent coupling is not the reference dual "
                             "controller; max_depth must be None/full horizon")
        max_depth = self.H
        nodes = list(tree.traverse())
        root = tree.nodes[tree.root]
        if root.x_o is None:
            raise ValueError("tree root carries no opponent state")
        x_o0 = tuple(float(v) for v in root.x_o)

        stats = dict(mode=cfg.mode, max_depth=max_depth, n_bin=0, n_rows=0,
                     n_nonlinear=0, gates_emitted=0, gates_reduced=0,
                     clips_emitted=0, clips_reduced=0, groups=0,
                     n_frozen_nodes=0)

        xb, vb = self._state_bounds(tree, x_o0)

        # Opponent lateral motion is deterministic; optimize longitudinal motion.
        for node in nodes:
            self.xo[node.id] = model.addVar(lb=xb[node.id][0], ub=xb[node.id][1],
                                            name=f"xo[{node.id}]")
            self.vxo[node.id] = model.addVar(lb=vb[node.id][0], ub=vb[node.id][1],
                                             name=f"vxo[{node.id}]")
            self.yo_const[node.id] = float(x_o0[1]) + float(x_o0[3]) * node.depth * dt
        model.addCons(self.xo[tree.root] == x_o0[0], name="opp_root_x")
        model.addCons(self.vxo[tree.root] == x_o0[2], name="opp_root_v")
        stats["n_rows"] += 2

        _root_safety = bool(self.geom.get("root_safety", False))
        for node in nodes:
            if node.parent is not None or _root_safety:
                self.slack_safe[node.id] = model.addVar(
                    lb=0.0, name=f"slack_safe[{node.id}]"
                )

        if cfg.tighten_ego_bounds:
            for node in nodes:
                (xl, xh), (yl, yh) = self._ego_bounds(x0, node.depth)
                if np.isfinite(xl):
                    model.chgVarLb(self.X[node.id][0], xl)
                    model.chgVarUb(self.X[node.id][0], xh)
                if np.isfinite(yl):
                    model.chgVarLb(self.X[node.id][2], yl)
                    model.chgVarUb(self.X[node.id][2], yh)

        add_road_bounds(model, self.X, self.U, self.geom, dt)

        children_by_parent: Dict[int, List] = {}
        for node in nodes:
            if node.parent is not None:
                children_by_parent.setdefault(node.parent, []).append(node)

        for parent in nodes:
            kids = children_by_parent.get(parent.id)
            if not kids:
                continue
            (xe_lo, xe_hi), (ye_lo, ye_hi) = self._ego_bounds(x0, parent.depth)

            z, gs = encode_gate(model, self.X[parent.id][0], self.X[parent.id][2],
                                self.xo[parent.id], self.yo_const[parent.id],
                                xe_rng=(xe_lo, xe_hi), ye_rng=(ye_lo, ye_hi),
                                xo_rng=xb[parent.id], p=p, name=str(parent.id))
            stats["n_bin"] += gs["n_bin"]
            stats["n_rows"] += gs["n_rows"]
            stats["n_nonlinear"] += gs["n_nonlinear"]
            if gs.get("nl_row"):
                self.nl_rows[f"gate_def[{parent.id}]"] = gs["nl_row"]
                _, g2, xe, xo, ye, yo = gs["nl_row"]
                fs = factor("squared_distance", "eq", output=g2)
                add_linear(fs, g2, 1.0, role="output")
                add_square_diff(fs, xe, xo, -1.0)
                add_square(fs, ye, -1.0, offset=float(yo))
                self.factor_specs[f"gate_def[{parent.id}]"] = fs
            stats["gates_reduced" if gs["reduced"] else "gates_emitted"] += 1
            self.z[parent.id] = z

            pvl, pvh = vb[parent.id]
            ev = self._ego_speed_bounds(x0, parent.depth)
            cands = [Candidate("ego", (self.X[parent.id][0], xe_lo, xe_hi),
                      (self.X[parent.id][2], ye_lo, ye_hi),
                      (self.X[parent.id][1], max(0., ev[0]), ev[1]))]
            for j, path in enumerate(opp.leader_paths):
                state = path[min(parent.depth, len(path)-1)]
                cands.append(Candidate(f"bg{j}", float(state[0]), float(state[1]), float(state[2])))
            af, afr, follow_stats = encode_follow(model, xo=self.xo[parent.id],
                xo_rng=xb[parent.id], yo=self.yo_const[parent.id],
                vo=self.vxo[parent.id], vo_rng=vb[parent.id], candidates=cands,
                p=FollowParams.from_gap_params(opp.gap, p.a_max), name=str(parent.id),
                factor_specs=self.factor_specs)
            for key in ("n_bin", "n_rows", "n_nonlinear"):
                stats[key] += follow_stats[key]
            stats["leader_candidates"] = stats.get("leader_candidates", 0)+follow_stats["n_cand"]
            stats["leader_pruned"] = stats.get("leader_pruned", 0)+len(follow_stats["dropped"])
            # After Hb, dynamics use the belief mean; observations use the inherited type.
            for tag in ("cau", "agg"):
                sign = THETA_SIGN[tag]
                nm = f"{parent.id},{tag}"
                stats["groups"] += 1

                vref, vr, vs = encode_vref(model, z, self.X[parent.id][1],
                                           theta_sign=sign,
                                           vxe_rng=self._ego_speed_bounds(x0, parent.depth),
                                           p=p, name=nm)
                stats["n_rows"] += vs["n_rows"]

                a_lo = (vr[0] - pvh) / p.k_c
                a_hi = (vr[1] - pvl) / p.k_c
                araw = model.addVar(lb=a_lo, ub=a_hi, name=f"araw[{nm}]")
                model.addCons(araw == (vref - self.vxo[parent.id]) / p.k_c,
                              name=f"araw_def[{nm}]")
                stats["n_rows"] += 1

                ac, _, cs = encode_clip(model, araw, (a_lo, a_hi),
                                        a_max=p.a_max, name=nm)
                stats["n_bin"] += cs["n_bin"]
                stats["n_rows"] += cs["n_rows"]
                stats["clips_reduced" if cs["reduced"] else "clips_emitted"] += 1
                ac, _, ms = encode_min(model, ac, (-p.a_max, p.a_max), af, afr, name=nm)
                stats["n_bin"] += ms["n_bin"]
                stats["n_rows"] += ms["n_rows"]
                self.ac[(parent.id, tag)] = ac

        # Beliefs must exist before constructing belief-weighted opponent dynamics.
        self._encode_belief(model, x0, tree, opp, Hb=hb)

        for node in nodes:
            if node.parent is None:
                continue
            pid = node.parent
            parent = tree.nodes[pid]
            bins_before, rows_before = model.getNBinVars(), model.getNConss()
            factors_before = len(self.factor_specs)
            if parent.depth < hb:
                a_expr = self.ac[(pid, node.tag)]
            else:
                a_expr = sum(self.b_node[(pid, tag)] * self.ac[(pid, tag)]
                             for tag in ("cau", "agg"))
            if parent.depth >= hb:
                ae = model.addVar(lb=-p.a_max, ub=p.a_max, name=f"opp_aexp[{node.id}]")
                model.addCons(ae == a_expr, name=f"opp_aexp_def[{node.id}]")
                fs = factor("opponent_belief_dynamics", "eq", output=ae)
                add_linear(fs, ae)
                for tag in ("cau", "agg"):
                    add_product(fs, self.b_node[(pid, tag)], self.ac[(pid, tag)], -1.)
                self.factor_specs[f"opp_aexp_def[{node.id}]"] = fs
                a_expr = ae
            encode_stop_step(model, self.xo[pid], self.vxo[pid], a_expr,
                self.xo[node.id], self.vxo[node.id], dt=dt, vr=vb[pid],
                ar=(-p.a_max, p.a_max), name=str(node.id), specs=self.factor_specs)
            stats["n_bin"] += model.getNBinVars()-bins_before
            stats["n_rows"] += model.getNConss()-rows_before
            stats["n_nonlinear"] += len(self.factor_specs)-factors_before

        # Replace nominal focus-opponent safety rows with endogenous-state rows.
        d_tau = float(self.geom["d_tau"])
        d_n = float(self.geom["d_n"])
        # Root safety is optional; future focus rows are soft, background rows hard.
        root_safety = bool(self.geom.get("root_safety", False))
        focus_rows = ("region_sum", "front", "back", "left", "right")
        drop = {f"{kind}[{n.id}]" for n in nodes for kind in focus_rows}
        if not root_safety:
            for k in range(len(self.bg_paths)):
                drop.update({f"{kind}_bg{k}[{root.id}]"
                             for kind in ("region_sum", "front", "back",
                                          "left", "right")})
        for cons in list(model.getConss()):
            if cons.name in drop:
                model.delCons(cons)
        for node in nodes:
            if node.parent is None and not root_safety:
                continue
            r_tau = self.X[node.id][0] - self.xo[node.id]
            r_n = self.X[node.id][2] - self.yo_const[node.id]
            s = self.slack_safe[node.id]
            model.addCons(quicksum(self.gamma[(node.id, key)]
                                   for key in ("f", "b", "l", "r")) == 1,
                          name=f"region_sum[{node.id}]")
            model.addCons(r_tau + s >= d_tau - self.M * (1 - self.gamma[(node.id, "f")]),
                          name=f"front[{node.id}]")
            model.addCons(-r_tau + s >= d_tau - self.M * (1 - self.gamma[(node.id, "b")]),
                          name=f"back[{node.id}]")
            model.addCons(-r_n + s >= d_n - self.M * (1 - self.gamma[(node.id, "l")]),
                          name=f"left[{node.id}]")
            model.addCons(r_n + s >= d_n - self.M * (1 - self.gamma[(node.id, "r")]),
                          name=f"right[{node.id}]")

        self.coupling_stats = stats
        return model

    def _encode_belief(self, model, x0, tree, opp, *, Hb: int) -> None:
        """Replace fixed tree weights and costs with endogenous weights.

        Use cost_n >= stage_cost and t_n >= w_n * cost_n to avoid cubic terms.
        """
        root_belief = tree.nodes[tree.root].belief
        w, b, bst = encode_belief_tree(model, tree, self.ac,
                                       sigma=SIGMA_OBS, Hb=Hb,
                                       root_belief=root_belief,
                                       factor_specs=self.factor_specs)
        self.w_node, self.b_node = w, b
        bst["Hb_detected"] = Hb
        self.belief_stats = bst

        drop = {f"obj_node[{n.id}]" for n in tree.traverse()}
        for c in list(model.getConss()):
            if c.name in drop:
                model.delCons(c)

        by_name = {v.name: v for v in model.getVars()}
        Q = np.asarray(self.w.get("Q", [1.0, 0.0, 1.0, 0.0]), dtype=float)
        Rw = np.asarray(self.w.get("R", [0.1, 0.1]), dtype=float)
        Qf = np.asarray(self.w.get("Qf", Q), dtype=float)
        w_slack = float(self.w.get("slack", 0.0))
        lam_sw = float(self.w.get("lane_switch", 0.0))
        w_goal = float(self.w.get("w_goal", 0.0))
        w_center = float(self.w.get("center_weight", 0.0))
        if not np.isfinite(w_center) or w_center < 0:
            raise ValueError("center_weight must be finite and nonnegative")
        w_safe = float(self.w.get("safety_slack",
                                  self.geom.get("safety_slack", 100.0)))

        terms = []
        for node in tree.traverse():
            depth = node.depth
            Qw = Qf if depth == self.H else Q
            cost = self._quad_cost(self.X[node.id], self._ref_state(depth), Qw)
            center_error = None
            if w_center > 0:
                # Track the lane selected at this node, including future lane changes.
                center_error = model.addVar(lb=-model.infinity(),
                                             name=f"center_error[{node.id}]")
                center = (float(self.geom["center"][1]) + float(self.geom["R0"])
                          + float(self.geom["wl"]) * (self.lane_idx[node.id] + 0.5))
                model.addCons(center_error == self.X[node.id][2] - center,
                              name=f"center_error_def[{node.id}]")
                cost = cost + w_center * center_error * center_error
            if node.id in self.U:
                cost = cost + self._quad_cost(self.U[node.id],
                                              self._ref_control(depth), Rw)
            sl = by_name.get(f"slack_lat[{node.id}]")
            if w_slack > 0 and sl is not None:
                cost = cost + w_slack * sl
            if lam_sw > 0 and node.id in self.b_pos:
                cost = cost + lam_sw * (self.b_pos[node.id] + self.b_neg[node.id])
            ag = by_name.get(f"abs_goal[{node.id}]")
            if w_goal > 0 and ag is not None:
                cost = cost + w_goal * ag
            ss = self.slack_safe.get(node.id)
            if w_safe > 0 and ss is not None:
                cost = cost + w_safe * ss * ss

            c_n = model.addVar(lb=0.0, name=f"cost_n[{node.id}]")
            cost_name = f"cost_n_def[{node.id}]"
            model.addCons(c_n >= cost, name=cost_name)
            fs = factor("node_cost", "ge", output=c_n)
            add_linear(fs, c_n, 1.0, role="output")
            if center_error is not None:
                add_square(fs, center_error, -w_center, role="state")
            for i, q_i in enumerate(Qw):
                if q_i:
                    add_square(fs, self.X[node.id][i], -float(q_i),
                               offset=float(self._ref_state(depth)[i]), role="state")
            if node.id in self.U:
                for j, r_j in enumerate(Rw):
                    if r_j:
                        add_square(fs, self.U[node.id][j], -float(r_j),
                                   offset=float(self._ref_control(depth)[j]), role="control")
            if w_slack > 0 and sl is not None:
                add_linear(fs, sl, -w_slack, role="lane_slack")
            if lam_sw > 0 and node.id in self.b_pos:
                add_linear(fs, self.b_pos[node.id], -lam_sw, role="lane_change")
                add_linear(fs, self.b_neg[node.id], -lam_sw, role="lane_change")
            if w_goal > 0 and ag is not None:
                add_linear(fs, ag, -w_goal, role="goal_distance")
            if w_safe > 0 and ss is not None:
                add_square(fs, ss, -w_safe, role="safety_slack")
            self.factor_specs[cost_name] = fs
            t_n = model.addVar(lb=0.0, name=f"tw_n[{node.id}]")
            wn = w[node.id]
            obj_name = f"obj_w[{node.id}]"
            model.addCons(t_n >= (float(wn) if isinstance(wn, (int, float)) else wn) * c_n,
                          name=obj_name)
            fs = factor("weighted_objective", "ge", output=t_n)
            add_linear(fs, t_n, 1.0, role="output")
            add_product(fs, wn, c_n, -1.0, roles=("weight", "cost"))
            self.factor_specs[obj_name] = fs
            terms.append(t_n)

        model.setObjective(quicksum(terms), "minimize")


    def solved_opponent_path(self, model) -> Optional[Dict[int, Tuple[float, float]]]:
        """Return {node_id: (xo, vxo)} from the incumbent, or None."""
        if not self.xo or model.getNSols() == 0:
            return None
        out = {}
        for nid, var in self.xo.items():
            try:
                out[nid] = (float(model.getVal(var)), float(model.getVal(self.vxo[nid])))
            except Exception:
                return None
        return out
