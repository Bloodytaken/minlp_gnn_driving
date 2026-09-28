"""SCIP counterpart of opponent_model.traffic_follow.step_stop_aware (straight road)."""
from offline_gnn.factor_schema import factor, add_linear, add_square, add_product


def encode_stop_step(model, x, v, a, xn, vn, *, dt, vr, ar, name, specs):
    raw_lo, raw_hi = vr[0]+dt*ar[0], vr[1]+dt*ar[1]
    if raw_lo >= 0:
        model.addCons(xn == x+dt*v+.5*dt*dt*a, name=f"opp_dyn_x[{name}]")
        model.addCons(vn == v+dt*a, name=f"opp_dyn_v[{name}]")
        return
    stop = model.addVar(vtype="B", name=f"opp_stop[{name}]")
    Mv = max(abs(raw_lo), abs(raw_hi))+1
    raw = v+dt*a
    model.addCons(raw <= Mv*(1-stop), name=f"stop_sign_hi[{name}]")
    model.addCons(raw >= -Mv*stop, name=f"stop_sign_lo[{name}]")
    model.addCons(vn <= Mv*(1-stop), name=f"stop_speed_zero[{name}]")
    model.addCons(vn-raw <= Mv*stop, name=f"stop_speed_hi[{name}]")
    model.addCons(vn-raw >= -Mv*stop, name=f"stop_speed_lo[{name}]")
    dxmax = max(0., vr[1])*dt+.5*dt*dt*max(0., ar[1])
    dx = model.addVar(lb=0., ub=dxmax+1e-6, name=f"opp_dx[{name}]")
    model.addCons(xn == x+dx, name=f"opp_dyn_x[{name}]")
    Mx = dxmax+dt*max(abs(vr[0]), abs(vr[1]))+.5*dt*dt*max(abs(ar[0]), abs(ar[1]))+1
    model.addCons(dx-dt*v-.5*dt*dt*a <= Mx*stop, name=f"stop_pos_hi[{name}]")
    model.addCons(dx-dt*v-.5*dt*dt*a >= -Mx*stop, name=f"stop_pos_lo[{name}]")
    # If stopping: distance = v^2/(-2a). Division-free and only quadratic.
    Mr = 2*max(abs(ar[0]), abs(ar[1]))*(dxmax+1e-6)+max(abs(vr[0]), abs(vr[1]))**2+1
    for suffix, sense, sign in (("hi", "le", 1.), ("lo", "ge", -1.)):
        residual = 2*a*dx+v*v+sign*Mr*stop-sign*Mr
        cname = f"stop_distance_{suffix}[{name}]"
        model.addCons(residual <= 0 if sign > 0 else residual >= 0, name=cname)
        fs = factor("opponent_belief_dynamics", sense, constant=-sign*Mr)
        add_product(fs, a, dx, 2); add_square(fs, v)
        add_linear(fs, stop, sign*Mr)
        specs[cname] = fs
