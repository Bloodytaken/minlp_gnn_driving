"""Shared highway traffic law and simultaneous nominal forecasts.

The focus model has endogenous ego/follower states and given paths for every
other vehicle. Collision-constraint selection (max_bg) never limits leaders.
Observation noise belongs to the commanded-acceleration measurement, not the dynamics.
"""
from dataclasses import dataclass, field, replace
import math
import numpy as np

from .opponent import OpponentModel, StraightRoadGeometry
from .traffic_follow import GapParams, find_leader, follow_bound

MODEL_VERSION = "leader_cap_v1_sensor_noise"


@dataclass
class FollowingOpponentModel(OpponentModel):
    geometry: object = field(default_factory=StraightRoadGeometry)
    gap: GapParams = field(default_factory=GapParams)
    leader_paths: tuple = ()

    def mean_accel_scalar(self, x_o, x_e, theta_sign, *, v_star=None, depth=0):
        others = [path[min(depth, len(path)-1)] for path in self.leader_paths]
        return self.mean_with_traffic(x_o, x_e, theta_sign, others, v_star=v_star)

    def mean_with_traffic(self, x_o, x_e, theta_sign, others, *, v_star=None):
        a = super().mean_accel_scalar(x_o, x_e, theta_sign, v_star=v_star)
        lead = find_leader(x_o, [x_e, *others], self.geometry, self.gap)
        if lead is not None:
            _, gap, speed = lead
            a = min(a, follow_bound(x_o[2], speed, gap,
                    a_ref=self.params.a_max, v_star=self.params.v_star, p=self.gap))
        return float(np.clip(a, -self.params.a_max, self.params.a_max))

    def for_vehicle(self, vehicle, paths):
        cruise = getattr(vehicle, "cruise_v", None) if vehicle is not None else None
        params = replace(self.params, v_star=float(cruise)) if cruise is not None else self.params
        return replace(self, params=params, leader_paths=tuple(
            tuple(tuple(s) for s in path) for vid, path in paths.items()
            if vehicle is None or vid != vehicle.vid))


def following_model(model=None):
    if isinstance(model, FollowingOpponentModel):
        return model
    if model is None:
        return FollowingOpponentModel()
    if not isinstance(model.geometry, StraightRoadGeometry):
        raise ValueError("MINLP following model requires straight-road geometry")
    return FollowingOpponentModel(geometry=model.geometry, params=model.params)
