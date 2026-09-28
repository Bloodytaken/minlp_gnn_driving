"""Straight-road geometry, dynamics and cost configuration."""

import math
import numpy as np

def minlp_to_tree_state(x):
    arr = np.asarray(x, dtype=float).reshape(-1)
    return np.array([arr[0], arr[2], arr[1], arr[3]], dtype=float)


def tree_to_minlp_state(x):
    arr = np.asarray(x, dtype=float).reshape(-1)
    return np.array([arr[0], arr[2], arr[1], arr[3]], dtype=float)


def straight_road_geom(H=6, M=2, lane0=0, wl=3.5, d_tau=None, d_n=None,
                       lon_margin=1.5, lat_margin=0.5, legacy_box=False):
    """Road geometry with center-to-center longitudinal/lateral margins.
    Defaults include both vehicle footprints and the requested buffers.
    The optional legacy box mode uses the former half-footprint margins.
    """
    center = np.array([0.0, 0.0])
    R0 = 0.0
    xs = np.linspace(0.0, 60.0, H + 1)
    ref_path = [(float(x), 0.0) for x in xs]
    vehicle_length, vehicle_width = 4.5, 2.2
    if legacy_box:
        dt_def, dn_def = vehicle_length / 2 + 1.5, vehicle_width / 2 + 0.9
    else:
        dt_def, dn_def = vehicle_length + lon_margin, vehicle_width + lat_margin
    dn_val = float(d_n) if d_n is not None else dn_def
    if dn_val >= wl:
        raise ValueError(
            f"d_n={dn_val:.2f} >= lane width {wl}: a vehicle in an adjacent lane "
            f"could never satisfy the left/right regions, so no lane change is "
            f"ever feasible. Lower lat_margin (max {wl - vehicle_width:.2f}).")
    return {
        "center": center,
        "R0": R0,
        "wl": wl,
        "M": M,
        "lane0": lane0,
        "ref_path": ref_path,
        "vehicle_length": vehicle_length,
        "vehicle_width": vehicle_width,
        "d_tau": float(d_tau) if d_tau is not None else dt_def,
        "d_n":   dn_val,
        "legacy_box": bool(legacy_box),
        "u_min": [-2.5, -1.5],
        "u_max": [ 2.5,  1.5],
        "v_min": [ 0.0, -1.5],
        "v_max": [15.0,  1.5],
        "is_straight_road": True,
    }


def make_roll_dynamics(dt, center, force_straight=True):
    center = np.asarray(center, dtype=float)
    dt2 = dt * dt
    def f_e(x_e, u_e):
        x, y, vx, vy = map(float, x_e)
        ax = float(u_e[0]) if u_e is not None and len(u_e) > 0 else 0.0
        ay = float(u_e[1]) if u_e is not None and len(u_e) > 1 else 0.0
        return (x + vx*dt + 0.5*ax*dt2, y + vy*dt + 0.5*ay*dt2,
                vx + ax*dt, vy + ay*dt)
    def f_o(x_o, u_tau):
        x, y, vx, vy = x_o
        if force_straight:
            ax, ay = float(u_tau), 0.0
        else:
            theta = math.atan2(y - center[1], x - center[0])
            tau = np.array([math.sin(theta), -math.cos(theta)])
            ax, ay = float(u_tau) * tau[0], float(u_tau) * tau[1]
        return (x + vx*dt + 0.5*ax*dt2, y + vy*dt + 0.5*ay*dt2,
                vx + ax*dt, vy + ay*dt)
    return f_e, f_o


def make_weights(H, x_goal, geom, current_lane=0, v_nom=8.0, u_ref=None):
    # x_ref tracks current lane center; Q[2]=0 so no absolute-y cost.
    # Lane-change incentive comes solely from w_goal on the integer lane_idx variable.
    Q  = [0.0, 1.0, 0.0, 5.0]
    Qf = [0.0, 2.0, 0.0, 2.0]
    R  = [0.05, 0.1]

    wl       = geom["wl"]
    center_y = geom["center"][1]
    R0       = geom["R0"]
    lane_center_y = center_y + (R0 + wl * (current_lane + 0.5))

    goal_lane_y = float(x_goal[1])
    goal_lane = int(np.clip(round((goal_lane_y - center_y - R0) / wl - 0.5),
                            0, geom["M"] - 1))

    w = {
        "Q": Q, "Qf": Qf, "R": R,
        "slack": 30.0,
        "lane_switch": 0.0,
        "goal_lane": goal_lane,
        "w_goal": 100.0,
        "x_ref": lambda depth: np.array([0.0, v_nom, lane_center_y, 0.0], dtype=float),
    }
    if u_ref is not None:
        w["u_ref"] = u_ref
    return w
