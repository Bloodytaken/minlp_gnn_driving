"""Numeric leader selection and car-following dynamics used by the MINLP."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

State = Tuple[float, float, float, float]


class _Geom:
    """Minimal tangent-frame protocol: anything with `tangent_frame(p)`."""


@dataclass
class GapParams:
    """IDM gap parameters. Values are the textbook defaults; `vehicle_length`
    matches `straight_road_geom` so `s` is a bumper-to-bumper gap, which is
    what IDM's s0/T are calibrated for."""

    s0: float = 2.0             # standstill distance (m)
    T: float = 1.0              # desired time headway (s)
    b: float = 2.0              # comfortable deceleration (m/s^2)
    delta: float = 4.0          # free-road exponent
    vehicle_length: float = 4.5
    w_lane: float = 3.5         # lane width, kept for callers that need it
    # Lateral reach of the leader test. It is a FOOTPRINT OVERLAP question, not
    # a lane-membership one: two 2.2 m bodies still overlap out to |dn| = 2.2 m,
    # while the old |dn| <= w_lane/2 = 1.75 m stopped looking at 1.75, leaving a
    # 0.45 m band in which a vehicle mid-lane-change was never braked for --
    # exactly the vehicle most likely to be hit.
    vehicle_width: float = 2.2
    # Footprint half-width plus 0.4 m relevance margin. This is not an
    # inter-sample collision guarantee (particularly at larger dt).
    w_follow: Optional[float] = 2.6
    # Deceleration scale of the car-following BOUND (see `follow_bound`). It is
    # the control limit, not the IDM's comfortable a_ref: the bound has to be
    # non-restrictive at a comfortable gap, and a 0.7 m/s^2 ceiling on a law
    # that works at +-4 would not be.
    a_bound: Optional[float] = 4.0
    look: float = 60.0          # ignore leaders beyond this; (s*/s)^2 is ~0 there

    def follow_halfwidth(self) -> float:
        return float(self.vehicle_width if self.w_follow is None else self.w_follow)






def follow_bound(v: float, v_lead: float, s: float, *, a_ref: float,
                 v_star: float, p: GapParams) -> float:
    """CAR-FOLLOWING BOUND: the largest acceleration the leader permits.

    This is an IDM-headway following term against that leader, used as an UPPER
    BOUND on whatever the theta branch wants:

        a = clip(min(a_theta, a_follow), a_min, a_max)

    Not the additive `min(0, .)` override it replaces. Additively, a theta
    branch commanding +4 against a bound of -1 still accelerated at +3 -- the
    vehicle could out-accelerate its own braking requirement. Under a bound it
    cannot: once the leader asks for -1, -1 is the most the vehicle may do.

    It is NOT a safety bound and must not be described as one. It says "do not
    accelerate past what following this leader allows", which with finite
    braking does not imply collision-freedom: from a state where the required
    stopping distance already exceeds the gap, every admissible acceleration
    leads to contact. That is a feasibility condition on the initial state, not
    a property this function can provide.

    Form and scale, both of which matter:

        s*       = s0 + max(0, v T + v dv / (2 sqrt(a_b b)))     (IDM headway)
        a_follow = a_b (1 - (s*/s)^2)

    The FREE-ROAD term of the IDM is deliberately absent. Including it makes the
    bound carry the leader's own desired-speed preference, and since a reactive
    vehicle tracking the ego routinely runs above the pool cruise speed
    (v = 11 against v_star = 8 gives 1 - (11/8)^4 = -2.6), the bound would bite
    at a 40 m gap and suppress theta on every frame with anything ahead. A
    following bound must say only "do not close on the leader", never "you are
    going faster than you like".

    The scale is the CONTROL limit a_b (default `params.a_max`), not the IDM's
    comfortable a_ref ~ 0.7. A bound scaled at 0.7 would cap a theta branch that
    works at +-4 down to +0.7 whenever any leader was in range, which is a
    restriction on free driving rather than on following. At a comfortable gap
    this form approaches +a_b; `min` leaves a_theta unchanged only when
    a_theta is below the bound. The same
    a_b appears inside s* so the headway and the bound agree about how hard the
    vehicle is willing to brake.
    """
    v = max(0.0, float(v))
    a_b = float(p.a_bound if p.a_bound is not None else a_ref)
    dv = v - float(v_lead)
    s_star = p.s0 + max(0.0, v * p.T + v * dv / (2.0 * math.sqrt(max(a_b, 1e-6) * p.b)))
    return float(a_b * (1.0 - (s_star / max(float(s), 1e-2)) ** 2))




def step_stop_aware(x: State, a: float, dt: float) -> State:
    """One tangential integration step that cannot reverse the vehicle.

    `advance_traffic` used to clamp the SPEED at zero while integrating the
    position with the full `v dt + a dt^2 / 2`, so a hard brake moved the car
    BACKWARDS: at a 0.15 m gap the complete IDM returns -2410 m/s^2, which over
    one 0.2 s step put the vehicle 47.6 m behind where it started while
    reporting v = 0. The reactive branch went through `f_o`, which clamps
    neither, so its speed could go negative outright.

    Here the vehicle stops WHERE IT RUNS OUT OF SPEED: if v + a dt < 0 it covers
    v^2 / 2|a| and ends at rest. Above that, ordinary constant-acceleration
    integration, so nothing changes on any step that does not hit zero.
    """
    px, py, vx, vy = (float(q) for q in x)
    if a < 0.0 and vx + a * dt < 0.0:
        return (px + vx * vx / (2.0 * abs(a)), py + vy * dt, 0.0, vy)
    return (px + vx * dt + 0.5 * a * dt * dt, py + vy * dt,
            max(0.0, vx + a * dt), vy)


def find_leader(x_o: State, others: Sequence[Optional[State]],
                geometry: _Geom, p: GapParams
                ) -> Optional[Tuple[int, float, float]]:
    """Nearest same-lane vehicle AHEAD of `x_o`. Returns (idx, s, v_lead) with
    `s` the bumper-to-bumper gap, or None.

    `others` may include the ego -- that is the point: a vehicle behind the ego
    must brake for it. This is a leader lookup only; it never feeds the theta
    branch, so it cannot change which vehicle the opponent is "interacting
    with" in the belief sense.

    The lateral test is footprint overlap (`GapParams.follow_halfwidth`), not
    lane membership: a vehicle straddling the lane line is the one most likely
    to be hit and the least likely to pass a |dn| <= w_lane/2 test.
    """
    p_o = np.asarray(x_o[:2], dtype=float)
    tau, n = geometry.tangent_frame(p_o)
    best = None
    for idx, x_i in enumerate(others):
        if x_i is None:
            continue
        d = np.asarray(x_i[:2], dtype=float) - p_o
        ds, dn = float(d @ tau), float(d @ n)
        if ds <= 0.0 or ds > p.look or abs(dn) >= p.follow_halfwidth():
            continue
        if best is None or ds < best[1]:
            best = (idx, ds, float(np.asarray(x_i[2:4], dtype=float) @ tau))
    if best is None:
        return None
    idx, ds, v_lead = best
    return idx, max(ds - p.vehicle_length, 1e-2), v_lead
