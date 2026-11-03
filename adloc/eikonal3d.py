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
    f (nx, ny, nz), and its gradient (3, nx, ny, nz). As in ADTomo, the 8 nodes around the station start from the
    straight-ray time by Simpson's rule; as in eikonal2d.station_table, they are not frozen."""
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
    u = eikonal_solve(u, 1.0 / f, h, verbose=False)
    return u, np.stack(np.gradient(u, h, edge_order=2))


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
    return {"station_loc": sta, "grid0": grid0, "shape": shape, "h": h,
            "tt": np.array([[t[0] for t in ts] for ts in tables], dtype=np.float32),  # (phase, station, nx, ny, nz)
            "grad_tt": np.array([[t[1] for t in ts] for ts in tables], dtype=np.float32)}  # (phase, station, 3, ...)


@njit
def _lookup(tt, grad, iph, ista, q, grid0, h):
    """Travel time and its gradient w.r.t. the event at q (n, 3) (km) from the tables [phase iph, station ista].

    Outside the grid, q_c is the nearest point on the edge and
        T(q) = T(q_c) + grad T(q_c) . (q - q_c),   grad T(q) = grad T(q_c),
    as in eikonal2d._lookup: the time is continuous across the edge and its gradient never zero.
    """
    n = q.shape[0]
    nn = np.array(tt.shape[2:])
    t = np.empty(n)
    g = np.empty((n, 3))
    for k in range(n):
        p, m = iph[k], ista[k]
        qc = np.empty(3)
        i0 = np.empty(3, np.int64)
        fr = np.empty(3)
        for d in range(3):
            qc[d] = min(max(q[k, d], grid0[d]), grid0[d] + (nn[d] - 1) * h)
            x = (qc[d] - grid0[d]) / h
            i0[d] = min(int(x), nn[d] - 2)
            fr[d] = x - i0[d]
        tq = 0.0
        gq = np.zeros(3)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    w = (fr[0] if a else 1 - fr[0]) * (fr[1] if b else 1 - fr[1]) * (fr[2] if c else 1 - fr[2])
                    i, j, l = i0[0] + a, i0[1] + b, i0[2] + c
                    tq += w * tt[p, m, i, j, l]
                    for d in range(3):
                        gq[d] += w * grad[p, m, d, i, j, l]
        for d in range(3):
            tq += gq[d] * (q[k, d] - qc[d])
            g[k, d] = gq[d]
        t[k] = tq
    return t, g


def _eval(event_index, station_index, phase_type, events, eikonal):
    """Travel times (n,) and gradients w.r.t. the event (n, 3) for events[event_index] at stations station_index."""
    ista = np.asarray(station_index).astype(np.int64)
    n = len(ista)
    q = np.ascontiguousarray(np.asarray(events, dtype=float)[np.broadcast_to(np.asarray(event_index), (n,))][:, :3])
    iph = np.broadcast_to(np.asarray(phase_type), (n,)).astype(np.int64)  # 0 (P) or 1 (S)
    return _lookup(eikonal["tt"], eikonal["grad_tt"], iph, ista, q, eikonal["grid0"], eikonal["h"])


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
        T, gT = eikonal["tt"][p, 0], eikonal["grad_tt"][p, 0]
        for name, A in ((f"t{ph}", [T[sta_ix, :, :], T[:, sta_iy, :], T[:, :, sta_iz]]),
                        (f"grad_t{ph}", [gT[0][sta_ix, :, :], gT[1][sta_ix, :, :], gT[2][sta_ix, :, :]])):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            for a, panel, title in zip(ax, A, ["x", "y", "z"]):
                fig.colorbar(a.pcolormesh(panel), ax=a)
                a.invert_yaxis()
                a.set_title(f"{name}_{title}")
            plt.savefig(f"{data_path}/slice_{name}_3d.png")
