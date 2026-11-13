import itertools
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

from .eikonal2d import MARGIN, MAX_ITER, TOL

###################################### Eikonal Solver ######################################

# |\nabla u| = f
# ((u - a1)^+)^2 + ((u - a2)^+)^2 + ((u - a3)^+)^2 = f^2 h^2


@njit(nogil=True)  # nogil: station tables are built in parallel threads
def calculate_unique_solution(a1, a2, a3, f, h):
    # a1, a2, a3 = np.sort([a1, a2, a3])
    if a1 > a2:
        a1, a2 = a2, a1
    if a2 > a3:
        a2, a3 = a3, a2
    if a1 > a2:
        a1, a2 = a2, a1

    x = a1 + f * h
    if x <= a2:
        return x
    else:
        B = -(a1 + a2)
        C = (a1**2 + a2**2 - f**2 * h**2) / 2
        x1 = (-B + np.sqrt(B**2 - 4 * C)) / 2.0
        x2 = (-B - np.sqrt(B**2 - 4 * C)) / 2.0
        if x1 > a2:
            x = x1
        else:
            x = x2
        if x <= a3:
            return x
        else:
            B = -2.0 * (a1 + a2 + a3) / 3.0
            C = (a1**2 + a2**2 + a3**2 - f**2 * h**2) / 3.0
            x1 = (-B + np.sqrt(B**2 - 4 * C)) / 2.0
            x2 = (-B - np.sqrt(B**2 - 4 * C)) / 2.0
            if x1 > a3:
                x = x1
            else:
                x = x2
        return x


@njit(nogil=True)
def sweeping_over_I_J_K(u, I, J, K, f, h):
    m = len(I)
    n = len(J)
    l = len(K)

    for i in I:
        for j in J:
            for k in K:
                if i == 0:
                    uxmin = u[i + 1, j, k]
                elif i == m - 1:
                    uxmin = u[i - 1, j, k]
                else:
                    uxmin = min(u[i - 1, j, k], u[i + 1, j, k])

                if j == 0:
                    uymin = u[i, j + 1, k]
                elif j == n - 1:
                    uymin = u[i, j - 1, k]
                else:
                    uymin = min(u[i, j - 1, k], u[i, j + 1, k])

                if k == 0:
                    uzmin = u[i, j, k + 1]
                elif k == l - 1:
                    uzmin = u[i, j, k - 1]
                else:
                    uzmin = min(u[i, j, k - 1], u[i, j, k + 1])

                u_new = calculate_unique_solution(uxmin, uymin, uzmin, f[i, j, k], h)

                u[i, j, k] = min(u_new, u[i, j, k])
    return u


@njit(nogil=True)
def sweeping(u, v, h):
    f = 1.0 / v  ## slowness

    m, n, l = u.shape
    I = np.arange(m)
    iI = I[::-1]
    J = np.arange(n)
    iJ = J[::-1]
    K = np.arange(l)
    iK = K[::-1]

    u = sweeping_over_I_J_K(u, I, J, K, f, h)
    u = sweeping_over_I_J_K(u, iI, J, K, f, h)
    u = sweeping_over_I_J_K(u, iI, iJ, K, f, h)
    u = sweeping_over_I_J_K(u, I, iJ, K, f, h)

    u = sweeping_over_I_J_K(u, I, iJ, iK, f, h)
    u = sweeping_over_I_J_K(u, I, J, iK, f, h)
    u = sweeping_over_I_J_K(u, iI, J, iK, f, h)
    u = sweeping_over_I_J_K(u, iI, iJ, iK, f, h)

    return u


def eikonal_solve(u, v, h, verbose=True):
    """Fast sweeping until no node changes by more than TOL (s), at most MAX_ITER iterations (as in eikonal2d)."""
    if verbose:
        print("Eikonal Solver: ")
    t0 = time.time()
    for i in range(MAX_ITER):
        u_old = np.copy(u)
        u = sweeping(u, v, h)

        err = np.max(np.abs(u - u_old))
        if verbose:
            print(f"Iteration {i}, Error = {err}")
        if err < TOL:
            break
    else:
        print(f"Warning: eikonal sweeps did not converge in {MAX_ITER} iterations")
    if verbose:
        print(f"Time: {time.time() - t0:.3f}")
    return u


###################################### Traveltime based on Eikonal Timetable ######################################
def station_table(f, grid0, h, sta):
    """Travel time T (s) of a station at sta = (x, y, z) (km) on the grid of first node grid0 and step h, with slowness
    f (nx, ny, nz). As in ADTomo, the 8 nodes around the station start from the straight-ray time by Simpson's rule;
    as in eikonal2d.station_table, they are not frozen."""
    shape = np.array(f.shape)
    p = (np.asarray(sta, dtype=float) - grid0) / h  # fractional grid coordinates
    c0 = np.clip(np.floor(p).astype(int), 0, shape - 2)  # the station's cell, inside the grid (as in ADTomo)
    offsets = np.array(list(itertools.product((0, 1), repeat=3)))
    corners = c0 + offsets
    fc = f[tuple(corners.T)]
    w = np.prod(np.where(offsets == 1, p - c0, 1 - (p - c0)), axis=1)  # trilinear weights at the station
    s_sta, s_mid = w @ fc, fc.mean()
    u = np.full(f.shape, 1000.0)
    u[tuple(corners.T)] = np.linalg.norm((corners - p) * h, axis=1) / 6.0 * (s_sta + 4.0 * s_mid + fc)
    return eikonal_solve(u, 1.0 / f, h, verbose=False)


def init_eikonal3d(config, stations):
    """One travel-time table per station on a 3-D grid shared by all stations, in absolute coordinates: x, y and z (km;
    z is depth, positive down, so a station above sea level has z < 0).

    config: h (km), xlim_km, ylim_km, zlim_km (the region and event depths), and vel {"P", "S"} in km/s, either a 1-D
    model {"Z", "P", "S"} read at absolute depth, on a grid that holds the region and the stations with MARGIN cells
    beyond (nodes on multiples of h, as in eikonal2d), or arrays (nx, ny, nz) on their own grid of first node grid0_km
    (as an ADTomo model). stations: x_km, y_km, z_km, in the order of the station indices. Events kept inside the region
    are looked up inside the grid; beyond it, see _lookup.
    """
    h = config["h"]
    sta = np.asarray(stations[["x_km", "y_km", "z_km"]], dtype=float)
    vel = config["vel"]
    if "Z" in vel:
        lo = np.minimum([config["xlim_km"][0], config["ylim_km"][0], config["zlim_km"][0]], sta.min(0))
        hi = np.maximum([config["xlim_km"][1], config["ylim_km"][1], config["zlim_km"][1]], sta.max(0))
        grid0 = (np.floor(lo / h) - MARGIN) * h
        shape = tuple(np.round(((np.ceil(hi / h) + MARGIN) * h - grid0) / h).astype(int) + 1)
        zgrid = grid0[2] + np.arange(shape[2]) * h
        vp, vs = [np.broadcast_to(np.interp(zgrid, vel["Z"], vel[ph]), shape) for ph in ("P", "S")]
    else:
        grid0, (vp, vs) = np.asarray(config["grid0_km"], dtype=float), (np.asarray(vel["P"]), np.asarray(vel["S"]))
        shape = vp.shape
        assert np.all(sta >= grid0) and np.all(sta <= grid0 + (np.array(shape) - 1) * h), "stations outside the velocity grid"

    t0 = time.time()
    with ThreadPoolExecutor() as pool:  # one table per thread; the sweeps release the GIL
        tables = [list(pool.map(lambda s: station_table(f, grid0, h, s), sta))
                  for f in [np.ascontiguousarray(1.0 / v, dtype=float) for v in (vp, vs)]]
    print(f"Eikonal: {len(sta)} station tables of {shape[0]} x {shape[1]} x {shape[2]} nodes in {time.time() - t0:.1f} s")
    # one node beyond each face, the linear continuation 2 T[0] - T[1], for the 4-node stencil of _lookup
    tt = np.pad(np.array(tables, dtype=np.float32), ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1)), mode="reflect", reflect_type="odd")
    return {"station_loc": sta, "grid0": grid0, "shape": shape, "h": h, "tt": tt}  # (phase, station, nx + 2, ny + 2, nz + 2)


@njit
def _weights(x, n):
    """Node i and the weights w, dw/dx of nodes i - 1 .. i + 2 at x (grid units) on an axis of n nodes: Catmull-Rom
    inside the grid, the edge's value and slope continued as a straight line beyond it."""
    xc = min(max(x, 0.0), n - 1.0)  # the nearest point of the grid
    i = min(int(xc), n - 2)
    f = xc - i
    f2, f3 = f * f, f * f * f
    w = ((-f3 + 2 * f2 - f) / 2, (3 * f3 - 5 * f2 + 2) / 2, (-3 * f3 + 4 * f2 + f) / 2, (f3 - f2) / 2)
    dw = ((-3 * f2 + 4 * f - 1) / 2, (9 * f2 - 10 * f) / 2, (-9 * f2 + 8 * f + 1) / 2, (3 * f2 - 2 * f) / 2)
    e = x - xc  # zero inside the grid
    return i, (w[0] + dw[0] * e, w[1] + dw[1] * e, w[2] + dw[2] * e, w[3] + dw[3] * e), dw


@njit
def _lookup(tt, iph, ista, q, grid0, h):
    """Travel time and its gradient w.r.t. the event at q (n, 3) (km) from the tables [phase iph, station ista]:

        T = sum_abc wx[a] wy[b] wz[c] T[a, b, c]      dT/dx = sum_abc dwx[a] wy[b] wz[c] T[a, b, c] / h

    with the 1-D weights of _weights: tricubic inside the grid, linear along each axis beyond it; T and grad T are
    continuous everywhere."""
    n = q.shape[0]
    nx, ny, nz = tt.shape[2] - 2, tt.shape[3] - 2, tt.shape[4] - 2
    t = np.empty(n)
    g = np.empty((n, 3))
    for k in range(n):
        p, m = iph[k], ista[k]
        i, wx, dwx = _weights((q[k, 0] - grid0[0]) / h, nx)
        j, wy, dwy = _weights((q[k, 1] - grid0[1]) / h, ny)
        l, wz, dwz = _weights((q[k, 2] - grid0[2]) / h, nz)
        tk, gx, gy, gz = 0.0, 0.0, 0.0, 0.0
        for a in range(4):
            for b in range(4):
                v, dv = 0.0, 0.0  # sum along z first (2x faster than the 64-term sum): with wz, and with dwz
                for c in range(4):
                    v += wz[c] * tt[p, m, i + a, j + b, l + c]  # node i - 1 is node i of the padded table
                    dv += dwz[c] * tt[p, m, i + a, j + b, l + c]
                tk += wx[a] * wy[b] * v
                gx += dwx[a] * wy[b] * v
                gy += wx[a] * dwy[b] * v
                gz += wx[a] * wy[b] * dv
        t[k] = tk
        g[k, 0], g[k, 1], g[k, 2] = gx / h, gy / h, gz / h
    return t, g


def _eval(event_index, station_index, phase_type, events, eikonal):
    """Travel times (n,) and gradients w.r.t. the event (n, 3) for events[event_index] at stations station_index."""
    ista = np.asarray(station_index).astype(np.int64)
    n = len(ista)
    q = np.ascontiguousarray(np.asarray(events, dtype=float)[np.broadcast_to(np.asarray(event_index), (n,))][:, :3])
    iph = np.broadcast_to(np.asarray(phase_type), (n,)).astype(np.int64)  # 0 (P) or 1 (S)
    return _lookup(eikonal["tt"], iph, ista, q, eikonal["grid0"], eikonal["h"])


def traveltime(event_index, station_index, phase_type, events, stations, eikonal):
    """
    event_index: [N,] or one index
    station_index: [N,]
    phase_type: [N,], 0 (P) or 1 (S)
    events: [M, 3]
    stations: [K, 3], unused: the station locations are those the tables were built for (eikonal["station_loc"])
    eikonal: dict from init_eikonal3d
    """
    return _eval(event_index, station_index, phase_type, events, eikonal)[0].astype(np.float32)


def grad_traveltime(event_index, station_index, phase_type, events, stations, eikonal):
    """Gradient (N, 3) of the travel times w.r.t. the event location; arguments as in traveltime."""
    return _eval(event_index, station_index, phase_type, events, eikonal)[1]


if __name__ == "__main__":
    # python -m adloc.eikonal3d: a homogeneous model, where the true time and gradient are known

    import os

    import matplotlib.pyplot as plt
    import pandas as pd

    data_path = "data"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    vel = {"p": 6.0, "s": 6.0 / 1.73}
    h = 1.0
    stations = pd.DataFrame({"x_km": [10.3], "y_km": [9.6], "z_km": [-1.2]})  # off the nodes, 1.2 km above sea level
    config = {"h": h, "xlim_km": [0, 20], "ylim_km": [0, 20], "zlim_km": [0, 20],
              "vel": {"Z": [0.0, 1.0], "P": [vel["p"]] * 2, "S": [vel["s"]] * 2}}
    eikonal = init_eikonal3d(config, stations)
    station_loc = stations[["x_km", "y_km", "z_km"]].values

    num_event = 10
    event_loc = np.random.rand(num_event, 3) * 20.0
    event_index = np.arange(num_event)
    station_index = np.zeros(num_event, dtype=int)
    phase_type = np.array([0] * (num_event // 2) + [1] * (num_event - num_event // 2))
    v = np.array([vel["p"] if x == 0 else vel["s"] for x in phase_type])
    t = np.linalg.norm(event_loc - station_loc, axis=-1) / v
    grad_t = (event_loc - station_loc) / np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=True) / v[:, np.newaxis]
    print(f"True traveltime: {t = }")
    print(f"True gradient: {grad_t = }")

    t = traveltime(event_index, station_index, phase_type, event_loc, station_loc, eikonal)
    grad_t = grad_traveltime(event_index, station_index, phase_type, event_loc, station_loc, eikonal)
    print(f"Computed traveltime: {t = }")
    print(f"Computed gradient: {grad_t = }")

    # slices through the station's nearest node of the table and of its gradient
    sta_ix, sta_iy, sta_iz = np.round((station_loc[0] - eikonal["grid0"]) / h).astype(int)
    for p, ph in enumerate(("p", "s")):
        T = eikonal["tt"][p, 0, 1:-1, 1:-1, 1:-1]
        gT = np.gradient(T, h)
        for name, A in ((f"t{ph}", [T[sta_ix, :, :], T[:, sta_iy, :], T[:, :, sta_iz]]),
                        (f"grad_t{ph}", [gT[0][sta_ix, :, :], gT[1][sta_ix, :, :], gT[2][sta_ix, :, :]])):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            for a, panel, title in zip(ax, A, ["x", "y", "z"]):
                fig.colorbar(a.pcolormesh(panel), ax=a)
                a.invert_yaxis()
                a.set_title(f"{name}_{title}")
            plt.savefig(f"{data_path}/slice_{name}_3d.png")

    # the gradient is the derivative of the returned time, inside the grid and beyond every face, edge and corner
    lo, hi = eikonal["grid0"], eikonal["grid0"] + (np.array(eikonal["shape"]) - 1) * h
    rng = np.random.default_rng(0)
    q = rng.uniform(lo - 3, hi + 3, (2000, 3))
    out = np.any((q < lo) | (q > hi), axis=1)
    i0 = np.zeros(len(q), int)
    g = _eval(np.arange(len(q)), i0, i0, q, eikonal)[1]
    fd = np.column_stack([(_eval(np.arange(len(q)), i0, i0, q + 1e-3 * d, eikonal)[0] - _eval(np.arange(len(q)), i0, i0, q - 1e-3 * d, eikonal)[0]) / 2e-3 for d in np.eye(3)])
    err = np.linalg.norm(g - fd, axis=1)
    print(f"gradient vs finite differences: max inside {err[~out].max():.1e}, outside {err[out].max():.1e} s/km")
    assert err.max() < 1e-5 and np.all(np.linalg.norm(g[out], axis=1) > 0.05)

    # time and gradient continuous across the node planes and the faces of the grid
    for d in range(3):
        for x in (lo[d], lo[d] + 5 * h, hi[d]):
            a = rng.uniform(lo + 1, hi - 1, (100, 3))
            a[:, d] = x - 1e-7
            b = a + 2e-7 * np.eye(3)[d]
            (ta, ga), (tb, gb) = [_eval(np.arange(100), i0[:100], i0[:100], e, eikonal) for e in (a, b)]
            assert np.abs(tb - ta).max() < 1e-5 and np.abs(gb - ga).max() < 1e-4
    print("time and gradient continuous across the nodes and the faces")

