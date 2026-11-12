import itertools
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

from .eikonal2d import MARGIN, MAX_ITER, TOL, _catmull_rom

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
    return {"station_loc": sta, "grid0": grid0, "shape": shape, "h": h,
            "tt": np.array(tables, dtype=np.float32)}  # (phase, station, nx, ny, nz)


@njit
def _stencil(x, n):
    """Along one axis of n nodes, for an event at x (grid units): the nearest grid point xc, the 4 nodes around it, and
    their Catmull-Rom weights and the weights' derivatives. A node one beyond the edge is the linear continuation
    2 T[0] - T[1] (or 2 T[n-1] - T[n-2]): its weight moves to those two nodes."""
    xc = min(max(x, 0.0), n - 1.0)
    i = min(int(xc), n - 2)
    w, d = _catmull_rom(xc - i)
    if i == 0:
        w, d = (0.0, w[1] + 2 * w[0], w[2] - w[0], w[3]), (0.0, d[1] + 2 * d[0], d[2] - d[0], d[3])
    if i == n - 2:
        w, d = (w[0], w[1] - w[3], w[2] + 2 * w[3], 0.0), (d[0], d[1] - d[3], d[2] + 2 * d[3], 0.0)
    return xc, (max(i - 1, 0), i, i + 1, min(i + 2, n - 1)), w, d


@njit
def _lookup(tt, iph, ista, q, grid0, h):
    """Travel time and its gradient w.r.t. the event at q (n, 3) (km) from the tables [phase iph, station ista]: the
    tricubic (Catmull-Rom) interpolation of the table and its exact derivative, a consistent pair, smooth across the
    nodes (a trilinear time has a kink at every node plane, where L-BFGS stops), as in eikonal2d._lookup and ADTomo.

    Outside the grid the time continues linearly from the nearest grid point c: T(c) + grad T(c) . (q - c). It is
    continuous, and its gradient (exact, with the cross terms where c moves along a face) is never zero, so an event
    pushed outside is pulled back."""
    n = q.shape[0]
    nx, ny, nz = tt.shape[2], tt.shape[3], tt.shape[4]
    t = np.empty(n)
    g = np.empty((n, 3))
    for k in range(n):
        p, m = iph[k], ista[k]
        x, y, z = (q[k, 0] - grid0[0]) / h, (q[k, 1] - grid0[1]) / h, (q[k, 2] - grid0[2]) / h  # event in grid units
        xc, ix, wx, dx = _stencil(x, nx)
        yc, iy, wy, dy = _stencil(y, ny)
        zc, iz, wz, dz = _stencil(z, nz)
        tq, tx, ty, tz, txy, txz, tyz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # T, its gradient and cross derivatives at c
        for a in range(4):
            for b in range(4):
                s, sz = 0.0, 0.0  # the sum along z, and its derivative in z
                for c in range(4):
                    v = tt[p, m, ix[a], iy[b], iz[c]]
                    s += wz[c] * v
                    sz += dz[c] * v
                tq += wx[a] * wy[b] * s
                tx += dx[a] * wy[b] * s
                ty += wx[a] * dy[b] * s
                tz += wx[a] * wy[b] * sz
                txy += dx[a] * dy[b] * s
                txz += dx[a] * wy[b] * sz
                tyz += wx[a] * dy[b] * sz
        # Outside the grid the time continues linearly from c with the slope at c:
        #     T(q) = T(c) + T_x(c) (x - xc) + T_y(c) (y - yc) + T_z(c) (z - zc)
        # (inside, q = c: the tricubic time). Its gradient by the chain rule: c follows the event along an axis where
        # the event is inside the grid (xc = x) and stays fixed along an axis where it is outside, so
        #     dT/dx = T_x(c) + T_xy(c) (y - yc) + T_xz(c) (z - zc)   x inside (c slides along the face)
        #     dT/dx = T_x(c)                                         x outside (c fixed in x)
        # and likewise for y and z, as in eikonal2d._lookup.
        ex, ey, ez = x - xc, y - yc, z - zc
        t[k] = tq + tx * ex + ty * ey + tz * ez
        g[k, 0] = (tx + (txy * ey + txz * ez if ex == 0.0 else 0.0)) / h
        g[k, 1] = (ty + (txy * ex + tyz * ez if ey == 0.0 else 0.0)) / h
        g[k, 2] = (tz + (txz * ex + tyz * ey if ez == 0.0 else 0.0)) / h
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
        T = eikonal["tt"][p, 0]
        gT = np.gradient(T, h)
        for name, A in ((f"t{ph}", [T[sta_ix, :, :], T[:, sta_iy, :], T[:, :, sta_iz]]),
                        (f"grad_t{ph}", [gT[0][sta_ix, :, :], gT[1][sta_ix, :, :], gT[2][sta_ix, :, :]])):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            for a, panel, title in zip(ax, A, ["x", "y", "z"]):
                fig.colorbar(a.pcolormesh(panel), ax=a)
                a.invert_yaxis()
                a.set_title(f"{name}_{title}")
            plt.savefig(f"{data_path}/slice_{name}_3d.png")
