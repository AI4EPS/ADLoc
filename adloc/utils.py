import copy
import multiprocessing as mp
import os
import pickle
import tempfile
import warnings
from contextlib import contextmanager, nullcontext

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from pyproj import Proj

from ._ransac import RANSACRegressor


OLD_NAMES = {"min_picks_ratio": "ransac_sample_ratio", "min_score": "ransac_min_r2"}  # names before 1.2


def translate_old_config(config):
    """Rename the settings from before 1.2 in place, so old configs keep running; one warning lists the changes.
    A config is old when it has min_picks_ratio or min_score, which every config before 1.2 had. Its min_p_picks and
    min_s_picks were score sums that each RANSAC sample had to reach; that rule is gone, so they
    are dropped. In a new config they are counts on the fitting picks.
    A new name, when also set, wins."""
    old = [k for k in OLD_NAMES if k in config]
    if not old:
        return config
    for k in old:
        config.setdefault(OLD_NAMES[k], config.pop(k))
    dropped = [k for k in ("min_p_picks", "min_s_picks") if config.pop(k, None) is not None]
    note = "; " + " and ".join(dropped) + " (sample score sums) are no longer used" if dropped else ""
    warnings.warn("ADLoc config names changed in 1.2: " + ", ".join(f"{k} -> {OLD_NAMES[k]}" for k in old) + note, stacklevel=3)
    return config


def invert(picks, stations, config, estimator, event_index, event_init):
    """
    Earthquake location for a single event using RANSAC.

    Two groups of settings. The event definition is GaMMA's, with GaMMA's defaults, tested here on the picks that fit
    the location (0 = off):
      min_picks                         their number (default 5)
      min_score_sum                     the sum of their scores (default 0.5 * min_picks)
      min_p_score_sum, min_s_score_sum  the same per phase (default 0.5 each)
      min_p_picks, min_s_picks          their number per phase (default 0)
    A pick fits when its residuals are within:
      max_residual_time                 travel time (default 1.0 s)
      max_residual_amplitude            amplitude (log10), with use_amplitude; None (default): no limit
      use_amplitude                     amplitudes always give the magnitude; True (default False) also uses them to
                                        judge: they enter R2, and max_residual_amplitude applies
    The sampling settings only decide which RANSAC trials are fitted; an event with no valid trial is skipped.
    A sample is drawn phase by phase (see _ransac.py): the number of P picks at random by the event's share of P,
    then the picks of each phase by score. A sample may hold no P or no S, so any pick can be left out as wrong.
      ransac_sample_ratio               a trial is located from max(min_picks, ransac_sample_ratio * picks) random picks (default 0.5)
      ransac_min_r2                     R2 of the trial's travel-time fit (default 0.5)
    """

    def is_model_valid(estimator, X, y):
        """
        X: idx_sta, type, score, amp
        """
        # n0 = np.sum(X[:, 1] == 0)  # P
        # n1 = np.sum(X[:, 1] == 1)  # S
        if estimator.events[0][2] < estimator.config["zlim_km"][0]:  # above the allowed depths (< 0 above sea level)
            return False

        if estimator.score(X, y) < config.get("ransac_min_r2", 0.5):
            return False

        return True

    MIN_PICKS = config.get("min_picks", 5)  # the defaults of the event definition are GaMMA's
    SAMPLE_RATIO = config.get("ransac_sample_ratio", 0.5)
    # amplitudes, when present, give the magnitude. use_amplitude (default False) also uses them to judge: they enter R2,
    # and picks beyond max_residual_amplitude (log10; None: no limit) do not fit. The same two settings as GaMMA.
    has_amp = "phase_amplitude" in picks.columns
    if has_amp:
        max_amp = config.get("max_residual_amplitude") if config.get("use_amplitude", False) else None
        MAX_RESIDUAL = [config.get("max_residual_time", 1.0), np.inf if max_amp is None else max_amp]
    else:
        MAX_RESIDUAL = config.get("max_residual_time", 1.0)

    if "station_term_time_p" not in stations.columns:
        stations["station_term_time_p"] = 0.0
    if "station_term_time_s" not in stations.columns:
        stations["station_term_time_s"] = 0.0
    if has_amp and ("station_term_amplitude" not in stations.columns):
        stations["station_term_amplitude"] = 0.0


    stations_ = stations[stations["station_id"].isin(picks["station_id"].unique())].copy()
    if "proj" not in config or config["proj"] is None:
        # global: a frame centred on this event's stations, each position counted once, as GaMMA's segments;
        # longitude averaged on the circle, so stations across +-180 deg stay together
        lon, lat = np.radians(np.unique(stations_[["longitude", "latitude"]].to_numpy(float), axis=0)).T
        lon0, lat0 = np.degrees(np.arctan2(np.sin(lon).mean(), np.cos(lon).mean())), np.degrees(lat.mean())
        proj = Proj(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0}  +units=km")
        # and its top no higher than this event's highest station, so one high station elsewhere does not lift it
        # (set for every event: the estimator is shared by the events of a worker)
        if "bfgs_bounds" not in config:
            z0, z1 = config["zlim_km"]
            estimator.set_params(config={**config, "zlim_km": (max(z0, stations_["z_km"].min()), z1)})
    else:
        proj = config["proj"]
    x_km, y_km = proj(stations_["longitude"].values, stations_["latitude"].values)
    stations_["x_km"] = x_km
    stations_["y_km"] = y_km

    if has_amp:
        X = picks.merge(
            # stations[["x_km", "y_km", "z_km", "station_id", "station_term_time", "station_term_amplitude"]],
            stations_[
                [
                    "x_km",
                    "y_km",
                    "z_km",
                    "station_id",
                    "station_term_time_p",
                    "station_term_time_s",
                    "station_term_amplitude",
                ]
            ],  ## Separate P and S station term
            on="station_id",
        )
    else:
        X = picks.merge(
            # stations[["x_km", "y_km", "z_km", "station_id", "station_term_time"]],
            stations_[
                ["x_km", "y_km", "z_km", "station_id", "station_term_time_p", "station_term_time_s"]
            ],  ## Separate P and S station term
            on="station_id",
        )
    t0 = X["phase_time"].min()

    z0 = min(5.0, np.mean(estimator.config["zlim_km"]))  # start depth in both modes, inside shallow search ranges
    if (event_init is None) or len(X) == 0:
        event_init = np.array([[np.median(X["x_km"]), np.median(X["y_km"]), z0, 0.0]])
    else:
        event_init = np.array([[event_init[0], event_init[1], z0, (event_init[3] - t0).total_seconds()]])

    # xstd = np.std(X["x_km"])
    # ystd = np.std(X["y_km"])
    # rstd = np.sqrt(xstd**2 + ystd**2)

    X.rename(columns={"phase_type": "type", "phase_score": "score", "phase_time": "t_s"}, inplace=True)
    X["t_s"] = (X["t_s"] - t0).dt.total_seconds()
    X["station_term_time"] = X.apply(
        lambda x: x["station_term_time_p"] if x["type"] == 0 else x["station_term_time_s"], axis=1
    )  ## Separate P and S station term
    X["t_s"] = X["t_s"] - X["station_term_time"]
    if has_amp:
        X.rename(columns={"phase_amplitude": "amp"}, inplace=True)
        X["amp"] = X["amp"] - X["station_term_amplitude"]

    estimator.set_params(**{"events": event_init})

    # ## Location using ADLoc
    # if has_amp:
    #     estimator.fit(X[["idx_sta", "type", "score", "amp"]].values, y=X[["t_s", "amp"]].values)
    # else:
    #     estimator.fit(X[["idx_sta", "type", "score"]].values, y=X[["t_s"]].values)
    # inlier_mask = np.ones(len(X)).astype(bool)

    ## Location using RANSAC
    num_picks = len(X)

    def is_event(x):
        """The event definition on a set of picks x (columns type: 0 P, 1 S; score)."""
        p_score, s_score = x["score"][x["type"] == 0], x["score"][x["type"] == 1]
        return (
            len(x) >= MIN_PICKS
            and x["score"].sum() >= config.get("min_score_sum", 0.5 * MIN_PICKS)
            and p_score.sum() >= config.get("min_p_score_sum", 0.5)
            and s_score.sum() >= config.get("min_s_score_sum", 0.5)
            and len(p_score) >= config.get("min_p_picks", 0)
            and len(s_score) >= config.get("min_s_picks", 0)
        )

    if not is_event(X):  # fitting only removes picks: an event that fails the definition now cannot meet it later
        picks["mask"] = 0
        picks["residual_time"] = 0.0
        if has_amp:
            picks["residual_amplitude"] = 0.0
        return picks, None

    reg = RANSACRegressor(
        estimator=estimator,
        random_state=0,
        min_samples=max(MIN_PICKS, int(SAMPLE_RATIO * num_picks)),
        # residual_threshold=MAX_RESIDUAL * (1.0 - np.exp(-rstd / 60.0)),  # not sure which one is better
        residual_threshold=MAX_RESIDUAL,
        is_model_valid=is_model_valid,
    )
    try:
        if has_amp:
            reg.fit(X[["idx_sta", "type", "score", "x_km", "y_km", "z_km"]].values, X[["t_s", "amp"]].values)
        else:
            reg.fit(X[["idx_sta", "type", "score", "x_km", "y_km", "z_km"]].values, X[["t_s"]].values)
    except Exception as e:
        # print(f"No valid model for event_index {event_index}.")
        message = "RANSAC could not find a valid consensus set."
        if str(e)[: len(message)] != message:
            print(e)
        picks["mask"] = 0
        picks["residual_time"] = 0.0
        if has_amp:
            picks["residual_amplitude"] = 0.0
        return picks, None

    estimator = reg.estimator_
    inlier_mask = reg.inlier_mask_

    ## Predict travel time
    output = estimator.predict(X[["idx_sta", "type", "score", "x_km", "y_km", "z_km"]].values)
    if has_amp:
        tt = output[:, 0]
        amp = output[:, 1]
    else:
        tt = output[:, 0]
    if has_amp:
        score = estimator.score(
            X[["idx_sta", "type", "score", "x_km", "y_km", "z_km"]].values[inlier_mask],
            y=X[["t_s", "amp"]].values[inlier_mask],
        )
    else:
        score = estimator.score(
            X[["idx_sta", "type", "score", "x_km", "y_km", "z_km"]].values[inlier_mask], y=X[["t_s"]].values[inlier_mask]
        )

    inliers = X[inlier_mask]  # the picks that fit the location
    if is_event(inliers):
        mean_residual_time = np.sum(np.abs(X["t_s"].values - tt) * inlier_mask) / np.sum(inlier_mask)
        if has_amp:
            mean_residual_amp = np.nanmean(np.abs(X["amp"].values - amp)[inlier_mask])  # over picks with an amplitude
        else:
            mean_residual_amp = 0.0
        x, y, z, t = estimator.events[0]
        if has_amp:
            mag = estimator.magnitudes[0]
        ## azimuthal gap: the largest angle around the epicentre with no station of a fitting pick
        dx, dy = (X[["x_km", "y_km"]].values[inlier_mask] - [x, y]).T
        az = np.sort(np.arctan2(dx, dy))  # station azimuths (rad)
        gap = np.degrees(np.diff(az, append=az[0] + 2 * np.pi).max())

        longitude, latitude = proj(x, y, inverse=True)

        event = {
            "idx_eve": event_index,  ## inside adloc, idx_eve is used which starts from 0 to N events
            "x_km": x,
            "y_km": y,
            "z_km": z,
            "longitude": longitude,
            "latitude": latitude,
            "depth_km": z,
            "time": t0 + pd.Timedelta(t, unit="s"),
            "adloc_score": score,
            "adloc_residual_time": mean_residual_time,
            "num_picks": np.sum(inlier_mask),
            "num_p_picks": int((inliers["type"] == 0).sum()),
            "num_s_picks": int((inliers["type"] == 1).sum()),
            "azimuthal_gap": gap,
        }
        if has_amp:
            event["magnitude"] = mag
            event["adloc_residual_amplitude"] = mean_residual_amp
        event = pd.DataFrame([event])
    else:
        inlier_mask = np.zeros(len(inlier_mask), dtype=bool)
        event = None

    picks["residual_time"] = X["t_s"].values - tt
    if has_amp:
        picks["residual_amplitude"] = X["amp"].values - amp
    else:
        picks["residual_amplitude"] = 0.0
    picks["mask"] = inlier_mask.astype(int)

    # ####################################### Debug #######################################
    # import os

    # import matplotlib.pyplot as plt

    # print(f"{max(MIN_PICKS, int(SAMPLE_RATIO * num_picks))=}")
    # print(f"{MAX_RESIDUAL=}")
    # fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(7, 4))
    # event = estimator.events[0]
    # X["dist"] = np.linalg.norm(X[["x_km", "y_km", "z_km"]].values - event[:3], axis=1)
    # ax[0, 0].scatter(X["dist"], X["t_s"], c="r", s=10, label="Picks")
    # ax[0, 0].scatter(X["dist"][~inlier_mask], X["t_s"][~inlier_mask], s=10, c="k", label="Noise")
    # ax[0, 0].scatter(X["dist"][inlier_mask], tt[inlier_mask], c="g", s=30, marker="x", label="Predicted")
    # ax[0, 0].set_xlabel("Distance (km)")
    # ax[0, 0].set_ylabel("Time (s)")
    # ax[0, 0].legend()
    # ax[0, 1].scatter(np.log10(X["dist"]), X["amp"], c="r", s=10, label="Picks")
    # ax[0, 1].scatter(np.log10(X["dist"][~inlier_mask]), X["amp"][~inlier_mask], s=10, c="k", label="Noise")
    # ax[0, 1].scatter(np.log10(X["dist"][inlier_mask]), amp[inlier_mask], c="g", s=30, marker="x", label="Predicted")
    # ax[0, 1].set_xlabel("Distance (km)")
    # ax[0, 1].set_ylabel("Amplitude")
    # if not os.path.exists("debug"):
    #     os.makedirs("debug")
    # fig.savefig(f"debug/event_{event_index}.png")
    # plt.close(fig)
    # raise
    # ####################################### Debug #######################################

    return picks, event


@contextmanager
def _one_thread_env():
    """One thread per math library (BLAS, OpenMP) in workers spawned inside: they read these at import, before
    threadpool_limits could reach them. The workers are the parallelism; threads inside them only contend."""
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update(dict.fromkeys(keys, "1"))
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_WORKER = None  # (config, estimator) in each worker, set once by _init_worker


def _init_worker(config, estimator, path):
    global _WORKER
    threadpool_limits(1)  # as _one_thread_env, for libraries already loaded
    if path is not None:  # the tables, shared through the page cache rather than copied
        eikonal = {**estimator.eikonal, "tt": np.load(path, mmap_mode="r")}
        swap = lambda d: {k: eikonal if v is estimator.eikonal else v for k, v in d.items()}
        estimator.config, config = swap(estimator.config), swap(config)
        estimator.eikonal = eikonal
    _WORKER = (config, estimator)


def _invert_job(picks, stations, event_index, event_init):
    config, estimator = _WORKER
    return invert(picks, stations, config, estimator, event_index, event_init)


def invert_location(picks, stations, config, estimator, events_init=None, iter=0):
    translate_old_config(config)  # here, so the warning comes once and not from every worker

    if "ncpu" in config:
        NCPU = config["ncpu"]
    else:
        NCPU = min(32, mp.cpu_count() - 1)  # as GaMMA: more workers gave no speed on 64 cores
    print(f"Using {NCPU} CPUs for ADLoc inversion.")

    jobs = []
    num_events = len(picks["idx_eve"].unique())
    events_inverted = []
    picks_inverted = []
    pbar = tqdm(total=num_events, desc=f"Iter {iter}")
    # config and the estimator reach each worker once, through the initializer; the tables (MBs) through a
    # memory-mapped file, since initializer data above the 64 KB pipe buffer make spawn start the workers one by one
    tmp = tempfile.TemporaryDirectory()
    shared = (config, estimator, None)
    eikonal = estimator.eikonal
    if eikonal is not None:
        path = os.path.join(tmp.name, "tt.npy")
        np.save(path, eikonal["tt"])
        light = {k: v for k, v in eikonal.items() if k != "tt"}
        est = copy.copy(estimator)
        est.eikonal, est.config = light, {k: light if v is eikonal else v for k, v in estimator.config.items()}
        shared = ({k: light if v is eikonal else v for k, v in config.items()}, est, path)
    assert len(pickle.dumps(shared)) < 2**16, "spawn blocks on initializer data above the 64 KB pipe buffer"
    with _one_thread_env():
        pool = mp.get_context("spawn").Pool(NCPU, initializer=_init_worker, initargs=shared)
    with tmp, pool:
        for idx_eve, picks_by_event in picks.groupby("idx_eve"):
            if config.get("proj") is None or events_init is None:
                event_init = None
            else:
                event_init = events_init[events_init["idx_eve"] == idx_eve]
                if len(event_init) == 0:
                    event_init = None
                else:
                    if len(event_init) > 1:
                        print(f"Multiple initial locations for event_index {idx_eve}.")
                    event_init = event_init[["x_km", "y_km", "z_km", "time"]].values[0]

            # picks_, event_ = invert(picks_by_event, stations, config, estimator, idx_eve, event_init)
            # if event_ is not None:
            #     events_inverted.append(event_)
            # picks_inverted.append(picks_)
            # pbar.update()

            job = pool.apply_async(
                _invert_job,
                (picks_by_event, stations, idx_eve, event_init),
                callback=lambda _: pbar.update(),
            )
            jobs.append(job)

        pool.close()
        pool.join()

        for job in jobs:
            picks_, event_ = job.get()
            if event_ is not None:
                events_inverted.append(event_)
            picks_inverted.append(picks_)

    if len(events_inverted) == 0:
        return None, None

    events_inverted = pd.concat(events_inverted, ignore_index=True)
    picks_inverted = pd.concat(picks_inverted)
    if events_init is not None:
        events_inverted = events_inverted.merge(events_init[["event_index", "idx_eve"]], on="idx_eve")

    print(f"ADLoc locates {len(events_inverted)} events outof {num_events} events")
    print(f"using {len(picks_inverted[picks_inverted['mask'] == 1])} picks outof {len(picks_inverted)} picks")

    return picks_inverted, events_inverted


def invert_location_iter(picks, stations, config, estimator, events_init=None, iter=0):
    translate_old_config(config)  # here, so the warning comes once and not from every worker

    events_inverted = []
    picks_inverted = []

    for idx_eve, picks_by_event in tqdm(picks.groupby("idx_eve"), desc=f"Iter {iter}"):
        if events_init is not None:
            event_init = events_init[events_init["idx_eve"] == idx_eve]
            if len(event_init) > 0:
                event_init = event_init[["x_km", "y_km", "z_km", "time"]].values[0]
        else:
            event_init = None

        picks_, event_ = invert(picks_by_event, stations, config, estimator, idx_eve, event_init)

        if event_ is not None:
            events_inverted.append(event_)
        picks_inverted.append(picks_)

    if len(events_inverted) == 0:
        return None, None
    events_inverted = pd.concat(events_inverted, ignore_index=True)
    picks_inverted = pd.concat(picks_inverted)
    if events_init is not None:
        events_inverted = events_inverted.merge(events_init[["event_index", "idx_eve"]], on="idx_eve")
    else:
        events_inverted["event_index"] = events_inverted["idx_eve"]

    print(f"ADLoc using {len(picks_inverted[picks_inverted['mask'] == 1])} picks outof {len(picks_inverted)} picks")
    return picks_inverted, events_inverted
