"""The event definition on the fitting picks, and RANSAC sampling rules that do not decide whether an event exists."""
import numpy as np
import pandas as pd
from pyproj import Proj

from adloc.gmpe import calc_amp
from adloc.sacloc2d import ADLoc
from adloc.utils import invert

PROJ = Proj("+proj=aeqd +lon_0=-122.8 +lat_0=38.8 +units=km")
VP, VS = 6.0, 6.0 / 1.73
EVENT = np.array([2.0, -1.0, 5.0])  # x, y, z (km)


def setup(n=12, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-20, 20, n), rng.uniform(-20, 20, n)
    lon, lat = PROJ(x, y, inverse=True)
    stations = pd.DataFrame({"station_id": [f"XX.S{i:02d}..HH" for i in range(n)], "longitude": lon, "latitude": lat, "z_km": 0.0, "idx_sta": range(n)})
    d = np.sqrt((x - EVENT[0]) ** 2 + (y - EVENT[1]) ** 2 + EVENT[2] ** 2)
    t0 = pd.Timestamp("2026-05-07T00:00:10")
    rows = [{"station_id": stations.station_id[i], "idx_sta": i, "phase_type": k, "phase_score": 0.9, "phase_time": t0 + pd.Timedelta(seconds=d[i] / v)}
            for i in range(n) for k, v in ((0, VP), (1, VS))]
    config = {"proj": PROJ, "xlim_km": (-30, 30), "ylim_km": (-30, 30), "zlim_km": (0, 20), "vel": {0: VP, 1: VS}}
    return pd.DataFrame(rows), stations, config, (x, y)


def locate(picks, stations, config):
    return invert(picks.copy(), stations.copy(), config, ADLoc(config), 0, None)


def test_event_is_located_with_its_quality_columns():
    picks, stations, config, (x, y) = setup()
    picks_out, event = locate(picks, stations, config)
    e = event.iloc[0]
    assert np.abs(e[["x_km", "y_km", "z_km"]].values.astype(float) - EVENT).max() < 0.2 and picks_out["mask"].all()
    assert (e.num_picks, e.num_p_picks, e.num_s_picks) == (24, 12, 12)
    az = np.sort(np.degrees(np.arctan2(x - EVENT[0], y - EVENT[1])))
    assert abs(e.azimuthal_gap - np.diff(np.append(az, az[0] + 360)).max()) < 2


def test_event_with_weak_s_picks_is_decided_by_the_definition():
    picks, stations, config, _ = setup()
    weak_s = pd.concat([picks[picks.phase_type == 0], picks[picks.phase_type == 1].iloc[:3].assign(phase_score=0.2)])  # 3 S, sum 0.6
    _, event = locate(weak_s, stations, config)  # S sum 0.6 >= min_s_score_sum 0.5
    assert event is not None and event.iloc[0].num_s_picks == 3
    _, event = locate(weak_s, stations, {**config, "min_s_score_sum": 1.0})  # the event definition does decide
    assert event is None


def test_event_with_one_phase_is_sampled():
    picks, stations, config, _ = setup()
    _, event = locate(picks[picks.phase_type == 0], stations, {**config, "min_s_score_sum": 0})  # P only, allowed by the definition
    assert event is not None and event.iloc[0].num_s_picks == 0


def test_one_wrong_pick_of_a_phase_is_left_out():
    """No rule forces a phase into every sample: then the only S pick, when wrong, could not be left out."""
    picks, stations, config, _ = setup()
    P, S = picks[picks.phase_type == 0], picks[picks.phase_type == 1]
    late = S.iloc[:1].assign(phase_time=S.phase_time.iloc[:1] + pd.Timedelta(seconds=5))
    for n in (5, 6, 8):
        _, event = locate(pd.concat([P.iloc[:n], late]), stations, {**config, "min_s_score_sum": 0})
        e = event.iloc[0]
        assert np.abs(e[["x_km", "y_km", "z_km"]].values.astype(float) - EVENT).max() < 0.2 and (e.num_p_picks, e.num_s_picks) == (n, 0)


def test_event_definition_on_the_fitting_picks():
    picks, stations, config, _ = setup()
    weak = picks.iloc[[0, 1, 6, 7, 12, 13]].assign(phase_score=0.31)  # P and S at 3 stations: scores sum to 1.86
    assert locate(weak, stations, config)[1] is None  # default min_score_sum 2.5
    assert locate(weak, stations, {**config, "min_score_sum": 0})[1] is not None
    assert locate(picks.iloc[:4], stations, config)[1] is None  # fewer than min_picks
    assert locate(picks, stations, {**config, "min_p_picks": 13})[1] is None


def test_unrelated_picks_are_not_an_event():
    picks, stations, config, _ = setup()
    rng = np.random.default_rng(1)
    noise = picks.assign(phase_time=pd.Timestamp("2026-05-07T00:00:10") + pd.to_timedelta(rng.uniform(0, 300, len(picks)), unit="s"))  # 24 picks over 5 minutes
    assert locate(noise, stations, config)[1] is None


def test_amplitudes_judge_only_with_use_amplitude_and_a_limit():
    picks, stations, config, _ = setup()
    x, y = PROJ(stations.longitude.values, stations.latitude.values)
    log_amp = calc_amp(2.0, EVENT[None], np.c_[x, y, stations.z_km.values])[:, 0]  # a magnitude 2 at each station
    amp = picks.assign(phase_amplitude=log_amp[picks.idx_sta.values])
    amp.loc[amp.index[0], "phase_amplitude"] += 5.0  # one pick 5 log10 units off
    for extra in ({"use_amplitude": False, "max_residual_amplitude": 1.0}, {"use_amplitude": True, "max_residual_amplitude": None}):
        kept, event = locate(amp, stations, {**config, **extra, "ransac_min_r2": 0})
        assert kept["mask"].all() and abs(event.iloc[0].magnitude - 2.0) < 0.5  # no pick rejected; the magnitude is still given
    kept, _ = locate(amp, stations, {**config, "use_amplitude": True, "max_residual_amplitude": 1.0, "ransac_min_r2": 0})
    assert kept["mask"].sum() == len(amp) - 1
