from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class OpponentModelParams:
    """
    Behavior model parameters for the roundabout opponent model.
    """

    d_int: float = 12.0  # interaction activation distance (meters)

    # Speed behavior
    k_c: float = 1.0  # proportional speed tracking gain
    delta_v: float = 1.0  # cautious/aggressive speed offset (Δv)
    v_star: float = 8.0  # cruising speed outside interaction zone (m/s)
    v_max: float = 20.0  # maximum allowed speed (m/s)

    # Input limits
    a_max: float = 4.0  # max allowed acceleration magnitude (m/s^2)

    # Noise (used for likelihood calculations / OU noise if desired)
    sigma: float = 0.25  # tangential Gaussian noise std
    ou_theta: float = 5.0  # OU noise decay
    dt_noise: float = 0.2  # noise integration step

    # Vehicle shape (for safety / footprint model)
    vehicle_width: float = 2.4

    # Debug/logging
    debug: bool = False

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """
        Deterministic prediction parameters (exclude stochastic terms).
        """
        params = asdict(cls())
        params.pop("sigma", None)
        params.pop("ou_theta", None)
        params.pop("dt_noise", None)
        return params

    @classmethod
    def get_all_params(cls) -> Dict[str, Any]:
        return asdict(cls())


def get_opponent_params() -> Dict[str, Any]:
    return OpponentModelParams.get_all_params()
