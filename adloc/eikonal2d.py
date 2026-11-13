import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

MAX_ITER, TOL = 100, 1e-8  # sweeps stop when no node changes by more than TOL (s), as in ADTomo and eikonal3d
MARGIN = 2  # grid cells beyond the events and stations, so their queries never touch the one-sided edge differences
DZ_STA = 0.1  # km: stations within 100 m of depth share a table (a station moves at most 50 m, ~0.01 s, a constant per
# station that station terms absorb), as in GaMMA; 10 m bins made ~200 tables and ~1 GB for a statewide network

###################################### Eikonal Solver ######################################

# |\nabla u| = f
# ((u - a1)^+)^2 + ((u - a2)^+)^2 + ((u - a3)^+)^2 = f^2 h^2


@njit(nogil=True)  # nogil: station tables are built in parallel threads
def calculate_unique_solution(a, b, f, h):
    d = abs(a - b)
    if d >= f * h:
        return min(a, b) + f * h
    else:
        return (a + b + np.sqrt(2.0 * f * f * h * h - (a - b) ** 2)) / 2.0


@njit(nogil=True)
def sweeping_over_I_J_K(u, I, J, f, h):
    m = len(I)
    n = len(J)

    for i in I:
        for j in J:
            if i == 0:
                uxmin = u[i + 1, j]
            elif i == m - 1:
                uxmin = u[i - 1, j]
            else:
                uxmin = min(u[i - 1, j], u[i + 1, j])

            if j == 0:
                uymin = u[i, j + 1]
            elif j == n - 1:
                uymin = u[i, j - 1]
            else:
                uymin = min(u[i, j - 1], u[i, j + 1])

            u_new = calculate_unique_solution(uxmin, uymin, f[i, j], h)

            u[i, j] = min(u_new, u[i, j])

    return u


@njit(nogil=True)
def sweeping(u, v, h):
    f = 1.0 / v  ## slowness

    m, n = u.shape
    I = np.arange(m)
    iI = I[::-1]
    J = np.arange(n)
    iJ = J[::-1]

    u = sweeping_over_I_J_K(u, I, J, f, h)
    u = sweeping_over_I_J_K(u, iI, J, f, h)
    u = sweeping_over_I_J_K(u, iI, iJ, f, h)
    u = sweeping_over_I_J_K(u, I, iJ, f, h)

    return u


def eikonal_solve(u, v, h, verbose=True):
    if verbose:
        print("Eikonal Solver: ")
    t0 = time.time()
    for i in range(MAX_ITER):
        u_old = np.copy(u)
        u = sweeping(u, v, h)

        err = np.max(np.abs(u - u_old))
        if verbose:
            print(f"Iter {i}, error = {err:.3f}")
        if err < TOL:
            break
    else:
        print(f"Warning: eikonal sweeps did not converge in {MAX_ITER} iterations")
    if verbose:
        print(f"Time: {time.time() - t0:.3f}")
    return u


###################################### Traveltime based on Eikonal Timetable ######################################
# def traveltime(event_loc, station_loc, phase_type, eikonal):
def traveltime(event_index, station_index, phase_type, events, stations, eikonal, vel={0: 6.0, 1: 6.0 / 1.73}):
    """
    event_index: list of event index
    station_index: list of station index
    phase_type: list of phase type
    events: list of event location
    stations: list of station location
    """
    if eikonal is None:
        v = np.array([vel[x] for x in phase_type])
        tt = np.linalg.norm(events[event_index] - stations[station_index], axis=-1, keepdims=False) / v
    else:
        if isinstance(event_index, int):
            event_index = np.array([event_index] * len(phase_type))
        tt = calc_traveltime(events[event_index], stations[station_index], phase_type, eikonal)

    return tt


@njit
def _node(tt, p, m, a, b, nr, nz):
    """Table value at node (a, b), one node beyond the grid at most: mirrored at r = 0 (T is even in r), linear
    continuation beyond the other edges."""
    a = abs(a)
    if a > nr - 1:
        return 2.0 * _node(tt, p, m, nr - 1, b, nr, nz) - _node(tt, p, m, nr - 2, b, nr, nz)
    if b < 0:
        return 2.0 * tt[p, m, a, 0] - tt[p, m, a, 1]
    if b > nz - 1:
        return 2.0 * tt[p, m, a, nz - 1] - tt[p, m, a, nz - 2]
    return tt[p, m, a, b]


@njit
def _catmull_rom(f):
    """Weights of nodes -1, 0, 1, 2 at fraction f in [0, 1], and their derivatives in f."""
    f2, f3 = f * f, f * f * f
    w = ((-f3 + 2 * f2 - f) / 2, (3 * f3 - 5 * f2 + 2) / 2, (-3 * f3 + 4 * f2 + f) / 2, (f3 - f2) / 2)
    d = ((-3 * f2 + 4 * f - 1) / 2, (9 * f2 - 10 * f) / 2, (-9 * f2 + 8 * f + 1) / 2, (3 * f2 - 2 * f) / 2)
    return w, d


@njit
def _bicubic(tt, iph, itab, r, z, z0, h):
    """Time and its exact derivatives (dt/dr, dt/dz) of the bicubic (Catmull-Rom) interpolant of tables tt (phase,
    table, nr, nz) on r = 0, h, ...: a consistent pair, smooth across the nodes (a bilinear time has a kink at every
    node, where L-BFGS stops: 24% of the Ridgecrest depths sat on a node), as in GaMMA. Beyond the grid the time
    continues linearly from the nearest grid point, with its exact, never-zero gradient."""
    n = r.shape[0]
    nr, nz = tt.shape[2], tt.shape[3]
    t, tr, tz = np.empty(n), np.empty(n), np.empty(n)
    for k in range(n):
        p, m = iph[k], itab[k]
        x, y = r[k] / h, (z[k] - z0) / h  # event in grid units (x = r / h, y = z from the top)
        xc, yc = min(x, nr - 1.0), min(max(y, 0.0), nz - 1.0)  # c: the nearest grid point (c = q inside the grid)
        i, j = min(int(xc), nr - 2), min(int(yc), nz - 2)
        wx, dwx = _catmull_rom(xc - i)
        wy, dwy = _catmull_rom(yc - j)
        tq, tx, ty, txy = 0.0, 0.0, 0.0, 0.0  # T, dT/dx, dT/dy, d2T/dxdy at c, in grid units
        for a in range(4):
            for b in range(4):
                v = _node(tt, p, m, i - 1 + a, j - 1 + b, nr, nz)
                tq += wx[a] * wy[b] * v
                tx += dwx[a] * wy[b] * v
                ty += wx[a] * dwy[b] * v
                txy += dwx[a] * dwy[b] * v
        # Outside the grid the time continues linearly from c with the slope at c:
        #     T(q) = T(c) + T_x(c) (x - xc) + T_y(c) (y - yc)
        # (inside, x = xc and y = yc: the bicubic time). Its gradient by the chain rule: c follows the event along an
        # axis where the event is inside the grid (xc = x) and stays fixed along an axis where it is outside, so
        #     dT/dx = T_x(c) + T_xy(c) (y - yc)   x inside (e.g. the event above the top: c slides along the top edge)
        #     dT/dx = T_x(c)                      x beyond rmax (c fixed in x)
        # and likewise for y. Time and gradient are continuous across the edge, and the gradient is the edge's, never
        # zero, so an event pushed outside is pulled back toward the data.
        t[k] = tq + tx * (x - xc) + ty * (y - yc)
        tr[k] = (tx + (txy * (y - yc) if x == xc else 0.0)) / h
        tz[k] = (ty + (txy * (x - xc) if y == yc else 0.0)) / h
    return t, tr, tz


def traveltime_grad(dx, dy, z, z_sta, phase_type, eikonal):
    """Travel times (n,) and their gradients w.r.t. the event location (n, 3), for events at horizontal offsets dx, dy
    (km) from stations at depth z_sta and at absolute depth z (km); phase_type 0 (P) or 1 (S), one value or one per pick."""
    n = len(dx)
    vec = lambda x, dtype: np.ascontiguousarray(np.broadcast_to(np.asarray(x), (n,)), dtype=dtype)
    z_sta, z_tab = vec(z_sta, float), eikonal["station_z"]
    itab = np.searchsorted((z_tab[1:] + z_tab[:-1]) / 2, z_sta)  # the nearest table by depth (robust to float32 depths)
    if np.any(np.abs(z_tab[itab] - z_sta) > DZ_STA / 2 + 1e-4):
        raise ValueError("a station depth has no eikonal table: pass every station's depth in config['station_z_km']")
    dx, dy = vec(dx, float), vec(dy, float)
    r = np.hypot(dx, dy)
    t, tr, tz = _bicubic(eikonal["tt"], vec(phase_type, np.int64), itab, r, vec(z, float), eikonal["zgrid"][0], eikonal["h"])
    c = (r > 1e-9) / np.where(r > 1e-9, r, 1.0)  # the horizontal direction (dx, dy) / r, and 0 on the station's axis
    return t, np.column_stack([tr * dx * c, tr * dy * c, tz])


def _eval(event_locs, station_locs, phase_type, eikonal):
    """Travel times and gradients for paired event and station locations ((n, 3) or (1, 3)), phases 0 (P) or 1 (S)."""
    e, s = np.broadcast_arrays(np.asarray(event_locs, float), np.asarray(station_locs, float))
    return traveltime_grad(e[:, 0] - s[:, 0], e[:, 1] - s[:, 1], e[:, 2], s[:, 2], phase_type, eikonal)


def calc_traveltime(event_locs, station_locs, phase_type, eikonal, vel={0: 6.0, 1: 6.0 / 1.73}):
    """
    event_locs: (num_event, 3) array of event locations
    station_locs: (num_station, 3) array of station locations
    phase_type: (num_event,) array of phase type, 0 (P) or 1 (S)
    eikonal: dictionary of eikonal solver, or None for straight rays at the velocities vel
    """

    if eikonal is None:
        v = np.array([vel[x] for x in phase_type])
        tt = np.linalg.norm(event_locs - station_locs, axis=-1, keepdims=False) / v
    else:
        tt = _eval(event_locs, station_locs, phase_type, eikonal)[0].astype(np.float32)

    return tt


def calc_grad_traveltime(event_loc, station_loc, phase_type, eikonal):
    """
    Gradient of the eikonal traveltime w.r.t. event location, computed directly from coordinates.
    event_loc: (num_event, 3) or (1, 3) array of event locations
    station_loc: (num_station, 3) array of station locations
    phase_type: (num_station,) array of phase type (0 for P, 1 for S)
    eikonal: dictionary of the eikonal timetable
    """
    return _eval(event_loc, station_loc, phase_type, eikonal)[1]


def grad_traveltime(event_index, station_index, phase_type, events, stations, eikonal):

    if isinstance(event_index, int):
        event_index = np.array([event_index] * len(phase_type))
    return calc_grad_traveltime(events[event_index], stations[station_index], phase_type, eikonal)


def station_table(zz, vel, z_sta, rgrid, zgrid, h):
    """Travel time T (s) of a station at r = 0 and depth z_sta (km) on the shared absolute (r, z) grid. As in ADTomo,
    the 4 nodes around the station start from the straight-ray time by Simpson's rule. Unlike ADTomo they are not frozen: a sharp contrast inside the station's cell can open a faster path
    to one (then the swept value is closer to the truth); a station exactly on a node also lets the first-order update
    0 + h * s lower its axial neighbour slightly (0.2 ms in a gradient model)."""
    f = np.tile(1.0 / np.interp(zgrid, zz, vel), (len(rgrid), 1))
    y = (z_sta - zgrid[0]) / h
    j0 = min(max(int(np.floor(y)), 0), len(zgrid) - 2)  # the station's cell, inside the grid (as in ADTomo)
    s_sta = (1 - (y - j0)) * f[0, j0] + (y - j0) * f[0, j0 + 1]
    s_mid = f[:2, j0 : j0 + 2].mean()
    u = np.full(f.shape, 1000.0)
    for i in (0, 1):
        for j in (j0, j0 + 1):
            u[i, j] = np.hypot(rgrid[i], zgrid[j] - z_sta) / 6.0 * (s_sta + 4.0 * s_mid + f[i, j])
    return eikonal_solve(u, 1.0 / f, h, verbose=False)


def init_eikonal2d(config):
    """One travel-time table per station depth on a grid shared by all stations: r (km from the station) by z (km,
    absolute depth, positive down; a station above sea level has z < 0). The 1-D model is read at absolute depth, so
    every station sees the same layers, and its top velocity extends upward to the highest station.

    config: vel {"Z", "P", "S"} (km, km/s), h (km), xlim_km and ylim_km (the region), zlim_km (the tables' depths:
    the events' zlim_km, or deeper, e.g. (0, 60), so far stations' first arrivals (Pn) are modelled), station_z_km
    (the station depths) and optionally station_xy_km (n, 2). Adds the tables and the grid to config and returns it.

    The grid holds, in r, the diagonal of the region and the stations together (as eikonal3d and GaMMA), and in z,
    zlim_km plus the stations, with MARGIN cells beyond, so events inside xlim, ylim and zlim are looked up inside the
    grid. Without station_xy_km, a station outside xlim, ylim (r > rmax) uses the linear extension of _bicubic.
    """
    h = config["h"]
    z_sta = np.unique(np.round(np.asarray(config["station_z_km"], dtype=float) / DZ_STA)) * DZ_STA
    lo, hi = np.array([config["xlim_km"], config["ylim_km"]], dtype=float).T  # (2,) each: x, y
    if "station_xy_km" in config:
        xy = np.asarray(config["station_xy_km"], dtype=float)
        lo, hi = np.minimum(lo, xy.min(0)), np.maximum(hi, xy.max(0))
    rmax = np.hypot(*(hi - lo))
    rgrid = np.arange(int(np.ceil(rmax / h)) + MARGIN + 1) * h
    z0 = (np.floor(min(config["zlim_km"][0], z_sta.min()) / h) - MARGIN) * h  # nodes on multiples of h, as in ADTomo
    z1 = (np.ceil(max(config["zlim_km"][1], z_sta.max()) / h) + MARGIN) * h
    zgrid = z0 + np.arange(int(round((z1 - z0) / h)) + 1) * h

    vel = config["vel"]
    zz = np.array(vel["Z"], dtype=float)
    t0 = time.time()
    with ThreadPoolExecutor() as pool:  # one table per thread; the sweeps release the GIL
        tables = [list(pool.map(lambda z: station_table(zz, np.array(vel[ph], dtype=float), z, rgrid, zgrid, h), z_sta))
                  for ph in ("P", "S")]
    out = {"station_z": z_sta, "rgrid": rgrid, "zgrid": zgrid, "nr": len(rgrid), "nz": len(zgrid), "h": h,
           "tt": np.array(tables, dtype=np.float32)}  # (phase, table, nr, nz)
    print(f"Eikonal: {len(z_sta)} station tables of {len(rgrid)} x {len(zgrid)} nodes in {time.time() - t0:.1f} s")
    return out  # only the tables and their grid, not the inputs: this dict goes to every worker


if __name__ == "__main__":

    import os

    import matplotlib.pyplot as plt

    data_path = "data"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    vel = {"p": 6.0, "s": 6.0 / 1.73}
    h = 1.0
    z_sta = -1.2  # off the nodes, 1.2 km above sea level
    config = {"vel": {"Z": [0.0, 1.0], "P": [vel["p"]] * 2, "S": [vel["s"]] * 2}, "h": h,
              "xlim_km": [0, 15], "ylim_km": [0, 15], "zlim_km": [0, 20], "station_z_km": [z_sta]}
    config = init_eikonal2d(config)

    num_event = 10
    event_loc = np.random.rand(num_event, 3) * np.array([15.0, 15.0, 20.0])
    event_index = np.arange(num_event)
    print(f"{event_loc = }")
    station_loc = np.array([0.0, 0.0, z_sta])
    print(f"{station_loc = }")
    station_loc = np.tile(station_loc, (num_event, 1))
    station_index = [0] * num_event
    # phase_type = np.random.choice(["p", "s"], num_event, replace=True)
    # print(f"{list(phase_type) = }")
    phase_type = np.array(["p"] * (num_event // 2) + ["s"] * (num_event - num_event // 2))
    v = np.array([vel[x] for x in phase_type])
    t = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=False) / v
    grad_t = (
        (event_loc - station_loc) / np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=True) / v[:, np.newaxis]
    )
    print(f"True traveltime: {t = }")
    print(f"True grad traveltime: {grad_t = }")

    # tp = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=False) / vel["p"]
    # print(f"{tp = }")
    # ts = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=False) / vel["s"]
    # print(f"{ts = }")

    mapping_int = {"p": 0, "s": 1}
    phase_type = np.array([mapping_int[x] for x in phase_type])
    # t = traveltime(event_loc, station_loc, phase_type, config)
    # grad_t = grad_traveltime(event_loc, station_loc, phase_type, config)
    t = traveltime(event_index, station_index, phase_type, event_loc, station_loc, config)
    grad_t = grad_traveltime(event_index, station_index, phase_type, event_loc, station_loc, config)
    print(f"Computed traveltime: {t = }")
    print(f"Computed grad traveltime: {grad_t = }")

    # the station's table and its gradient on the (r, z) grid
    for p, ph in enumerate(("p", "s")):
        plt.figure()
        plt.pcolormesh(config["tt"][p, 0])
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.savefig(f"{data_path}/slice_t{ph}_2d.png")

        g = np.gradient(config["tt"][p, 0], config["h"])
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for a, panel, title in zip(ax, g, ["r", "z"]):
            fig.colorbar(a.pcolormesh(panel), ax=a)
            a.invert_yaxis()
            a.set_title(f"grad_t{ph}_{title}")
        plt.savefig(f"{data_path}/slice_grad_t{ph}_2d.png")

    # checks with stations at elevation: a linear-gradient model v = v0 + g z has an exact time for any source and
    # receiver depth (the former tests/test_eikonal2d_topography.py)
    v0, gz_, h = 5.0, 0.04, 0.5
    elev = np.array([0.0, 1.0, 1.1, 1.23, 1.37, 1.5, 3.0])  # km above sea level; 1.0-1.5 sweeps one grid step
    exact = lambda r, z_ev, z_st: np.arccosh(1 + gz_**2 * (r**2 + (z_ev - z_st) ** 2) / (2 * (v0 + gz_ * z_ev) * (v0 + gz_ * z_st))) / gz_
    zz = np.array([-20.0, 100.0])
    eik = init_eikonal2d({"vel": {"Z": zz, "P": v0 + gz_ * zz, "S": (v0 + gz_ * zz) / 1.73}, "h": h, "xlim_km": [0, 60],
                          "ylim_km": [0, 60], "zlim_km": [-3, 30], "station_z_km": -elev})
    assert eik["tt"].shape[:2] == (2, len(elev)), "one table per station depth and phase"
    rng = np.random.default_rng(0)
    ev = np.column_stack([rng.uniform(1, 60, 2000), rng.uniform(0, 60, 2000), rng.uniform(0, 25, 2000)])
    p0 = np.zeros(len(ev), int)
    errs = np.array([calc_traveltime(ev, np.array([[0.0, 0.0, -e]]), p0, eik) - exact(np.hypot(ev[:, 0], ev[:, 1]), ev[:, 2], -e) for e in elev])
    spread = np.percentile(np.abs(errs - errs.mean(0)), 95)
    print(f"vs exact: max {np.abs(errs).max() * 1e3:.0f} ms (first-order bias), variation between stations p95 {spread * 1e3:.1f} ms")
    assert np.abs(errs).max() < 0.15 and spread < 0.01

    # the gradient is the derivative of the returned time, inside the grid and beyond every edge
    rmax, ztop, zbot = eik["rgrid"][-1], eik["zgrid"][0], eik["zgrid"][-1]
    st = np.array([[0.0, 0.0, -1.23]])
    rng = np.random.default_rng(1)
    k = 100  # beyond rmax, above the top, below the bottom, and the corners
    rr, zz_ = rng.uniform(1, rmax - 1, k), rng.uniform(ztop + 1, zbot - 1, k)
    rout, ztout, zbout = rmax + rng.uniform(0.1, 5, k), ztop - rng.uniform(0.1, 5, k), zbot + rng.uniform(0.1, 5, k)
    az = rng.uniform(0, 2 * np.pi, k)
    out = np.vstack([np.c_[r * np.cos(az), r * np.sin(az), z] for r, z in [(rout, zz_), (rr, ztout), (rr, zbout), (rout, ztout), (rout, zbout)]])
    q = np.vstack([ev[:300], out])
    pq = np.zeros(len(q), int)
    g = calc_grad_traveltime(q, st, pq, eik)
    fd = np.column_stack([(_eval(q + 1e-6 * d, st, pq, eik)[0] - _eval(q - 1e-6 * d, st, pq, eik)[0]) / 2e-6 for d in np.eye(3)])  # float64 times
    err = np.linalg.norm(g - fd, axis=1)
    print(f"gradient vs finite differences: max inside {err[:300].max():.1e}, outside {err[300:].max():.1e} s/km")
    assert err.max() < 1e-5 and np.all(np.linalg.norm(g[300:], axis=1) > 0.05)

    # time and gradient continuous across the edges; a station depth without a table raises
    for a, b in [([rmax - 1e-6, 0, 10], [rmax + 1e-6, 0, 10]), ([5, 5, ztop + 1e-6], [5, 5, ztop - 1e-6]), ([5, 5, zbot - 1e-6], [5, 5, zbot + 1e-6])]:
        t2, g2 = _eval(np.array([a, b], float), np.array([[0.0, 0.0, -1.5]]), np.array([0, 0]), eik)
        assert abs(t2[1] - t2[0]) < 1e-5 and np.abs(g2[1] - g2[0]).max() < 1e-4
    try:
        calc_traveltime(ev[:3], np.array([[0.0, 0.0, -0.7]]), [0, 0, 0], eik)
        raise AssertionError("expected a ValueError for a station depth without a table")
    except ValueError:
        pass
    print("all checks passed")
