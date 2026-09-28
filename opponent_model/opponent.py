"""
Roundabout opponent behavior model expressed directly in Cartesian space.

The construction mirrors the legacy implementation in
`roundabout_dual/agent_modeling/agent.py`, but exposes lightweight helpers that
only depend on the vehicle states `(x, y, v_x, v_y)` so that they can be used by
the new tree / belief modules without pulling in the full simulation stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
import math
import numpy as np

from parameters import OpponentModelParams


Vector = np.ndarray
State = Tuple[float, float, float, float]
ThetaSign = float  # cautious = -1, aggressive = +1


@dataclass
class RoundaboutGeometry:
    """
    Geometry description needed to extract tangential directions.

    `center` specifies the roundabout center (in Cartesian coordinates).  Every
    position is projected onto the tangential axis defined by the radial vector
    from `center` to `p`.
    """

    center: Tuple[float, float] = (0.0, 0.0)

    def tangent_frame(self, position: Vector) -> Tuple[Vector, Vector]:
        """
        Returns (tau, n) where tau is the unit tangential vector (directions of
        travel) and n is the outward normal at `position`.
        """
        cx, cy = self.center
        theta = math.atan2(position[1] - cy, position[0] - cx)
        tau = np.array([-math.sin(theta), math.cos(theta)], dtype=float)
        n = np.array([math.cos(theta), math.sin(theta)], dtype=float)
        return tau, n

# mixed_integer_dual_mpc/opponent.py (or a new helper module)
class StraightRoadGeometry(RoundaboutGeometry):
    def tangent_frame(self, position):
        tau = np.array([1.0, 0.0], dtype=float)   # along +x
        n   = np.array([0.0, 1.0], dtype=float)   # lateral +y
        return tau, n


@dataclass
class OpponentModel:
    """
    Reactive opponent behavioral model in Cartesian coordinates.

    - The deterministic part follows the Euclidean-distance switching rule from
      `roundabout_dual/agent_modeling/agent.py`: outside the interaction radius
      the opponent tracks a cruising speed `v_star`; inside, it reacts to the
      ego speed with an offset determined by `theta_sign` (±1 for cautious /
      aggressive).
    - Acceleration is defined along the tangential axis of the roundabout and
      can be converted back to Cartesian form.
    """

    geometry: RoundaboutGeometry = field(default_factory=RoundaboutGeometry)
    params: OpponentModelParams = field(default_factory=OpponentModelParams)

    def _split_state(self, state: State) -> Tuple[Vector, Vector]:
        if state is None:
            raise ValueError("State must be provided (x, y, vx, vy).")
        arr = np.asarray(state, dtype=float)
        if arr.shape[0] != 4:
            raise ValueError(f"State must have 4 elements, got {arr}")
        position = arr[:2]
        velocity = arr[2:]
        return position, velocity

    def _project_velocity(self, velocity: Vector, tau: Vector) -> float:
        return float(np.dot(velocity, tau))

    def mean_accel_scalar(self, x_o: State, x_e: State, theta_sign: ThetaSign,
                          *, v_star: Optional[float] = None) -> float:
        """
        Compute tangential acceleration command (scalar) for a given driver type.

        `v_star` overrides the CRUISE speed this vehicle falls back to outside
        the interaction ball. It exists so a vehicle can be given a latent type
        without also being given a new speed: `params.v_star` is one number for
        the whole pool, so promoting a slow idm vehicle to reactive would
        otherwise make it accelerate to the pool speed and destroy the traffic's
        speed spread. Inside `d_int` nothing changes -- the vehicle tracks the
        ego either way, which is what makes theta observable.
        """
        params = self.params
        p_o, v_o = self._split_state(x_o)
        p_e, v_e = self._split_state(x_e)

        tau, _ = self.geometry.tangent_frame(p_o)
        g = np.linalg.norm(p_e - p_o)
        v_hat_o = self._project_velocity(v_o, tau)
        v_hat_e = self._project_velocity(v_e, tau)

        if g <= params.d_int:
            v_ref = v_hat_e + theta_sign * params.delta_v
        else:
            v_ref = params.v_star if v_star is None else float(v_star)

        a_tau = (v_ref - v_hat_o) / params.k_c
        return float(np.clip(a_tau, -params.a_max, params.a_max))

    def mean_accel_vector(self, x_o: State, x_e: State, theta_sign: ThetaSign) -> Vector:
        """
        Cartesian acceleration corresponding to `mean_accel_scalar`.
        """
        a_tau = self.mean_accel_scalar(x_o, x_e, theta_sign)
        tau, _ = self.geometry.tangent_frame(self._split_state(x_o)[0])
        return a_tau * tau

    def sample_action(
        self,
        x_o: State,
        x_e: State,
        theta_sign: ThetaSign,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Draw a noisy tangential acceleration sample for branching.
        """
        mu = self.mean_accel_scalar(x_o, x_e, theta_sign)
        sigma = self.params.sigma
        if sigma <= 0:
            return mu
        noise = (rng or np.random.default_rng()).normal(scale=sigma)
        return float(np.clip(mu + noise, -self.params.a_max, self.params.a_max))

    @staticmethod
    def gaussian_likelihood(u_obs: float, mu: float, sigma: float) -> float:
        """
        Likelihood of observing `u_obs` given Gaussian noise on tangential accel.
        """
        if sigma <= 0:
            return 1.0 if abs(u_obs - mu) < 1e-12 else 0.0
        err = (u_obs - mu) / sigma
        return math.exp(-0.5 * err * err)

    def make_mu_functions(self, theta_tags: Tuple[str, ...]) -> dict[str, Callable[[State, State], float]]:
        """
        Convenience helper returning the `mu_fns` dictionary expected by
        `Tree.build_tree`: each entry maps a driver tag to the corresponding
        mean tangential acceleration function.
        """
        tag_to_sign = {
            "cau": -1.0,
            "agg": +1.0,
        }
        mu_fns = {}
        for tag in theta_tags:
            if tag not in tag_to_sign:
                raise KeyError(f"Unknown driver tag '{tag}'. Expected one of {tuple(tag_to_sign)}.")
            sign = tag_to_sign[tag]

            def _mu_fn(x_o: State, x_e: State, s=sign):
                return self.mean_accel_scalar(x_o, x_e, s)

            mu_fns[tag] = _mu_fn
        return mu_fns
