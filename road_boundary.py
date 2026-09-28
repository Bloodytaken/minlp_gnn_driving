"""Hard straight-road footprint bounds and a conservative braking corridor."""

def add_road_bounds(model, states, controls, geom, dt):
    if not geom.get("is_straight_road", False):
        return
    half_width = float(geom.get("vehicle_width", 2.2)) / 2
    lower = float(geom["center"][1]) + float(geom["R0"]) + half_width
    upper = lower + float(geom["M"]) * float(geom["wl"]) - 2 * half_width
    # |vy|^2/(2*a) <= vmax*|vy|/(2*a). This linear envelope leaves
    # enough space to brake laterally if the next solve has no incumbent.
    a_brake = min(-float(geom["u_min"][1]), float(geom["u_max"][1]))
    vmax = max(abs(float(geom["v_min"][1])), abs(float(geom["v_max"][1])))
    if a_brake <= 0 or upper <= lower:
        raise ValueError("road bounds require positive width and lateral braking")
    # If stopping takes less than one sample, fallback chooses -vy/dt;
    # its stopping displacement is then |vy|*dt/2 instead.
    stop_time_half = max(vmax / (2 * a_brake), float(dt) / 2)
    for nid, x in states.items():
        y, vy = x[2], x[3]
        model.addCons(y >= lower, name=f"road_lower[{nid}]")
        model.addCons(y <= upper, name=f"road_upper[{nid}]")
        model.addCons(y + stop_time_half * vy >= lower,
                      name=f"road_stop_lower[{nid}]")
        model.addCons(y + stop_time_half * vy <= upper,
                      name=f"road_stop_upper[{nid}]")
        if nid in controls:
            # Constant-acceleration y(t) is a quadratic Bezier curve with
            # control points y0, y0+dt*vy0/2, y1. Bounding all three bounds
            # the whole interval, including an extremum between samples.
            mid = y + float(dt) * vy / 2
            model.addCons(mid >= lower, name=f"road_mid_lower[{nid}]")
            model.addCons(mid <= upper, name=f"road_mid_upper[{nid}]")
