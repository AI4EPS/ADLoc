# %% Download test data
# !if [ -f demo.tar ]; then rm demo.tar; fi
# !if [ -d test_data ]; then rm -rf test_data; fi
# !wget -q https://github.com/AI4EPS/datasets/releases/download/test_data/test_data.tar
# !tar -xf test_data.tar

# %%
import json
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Proj

from adloc.eikonal2d import init_eikonal2d
from adloc.sacloc2d import ADLoc
from adloc.utils import invert_location, invert_location_iter
from utils import plotting_ransac

np.random.seed(42)

# %%
if __name__ == "__main__":
    # # %%
    # ##################################### DEMO DATA #####################################
    # # region = "synthetic"
    # region = "ridgecrest"
    # data_path = f"test_data/{region}/"
    # picks_file = os.path.join(data_path, "gamma_picks.csv")
    # events_file = os.path.join(data_path, "gamma_events.csv")
    # stations_file = os.path.join(data_path, "stations.csv")

    # picks = pd.read_csv(picks_file, parse_dates=["phase_time"])
    # events = pd.read_csv(events_file, parse_dates=["time"])
    # stations = pd.read_csv(stations_file)
    # events_init = events.copy()

    # # picks = pd.read_csv(os.path.join(data_path, "phasenet_plus_picks.csv"), parse_dates=["phase_time"])

    # config = json.load(open(os.path.join(data_path, "config.json")))
    # config["mindepth"] = 0.0
    # config["maxdepth"] = 30.0
    # config["use_amplitude"] = True

    # # ## Eikonal for 1D velocity model
    # zz = [0.0, 5.5, 16.0, 32.0]
    # vp = [5.5, 5.5, 6.7, 7.8]
    # vp_vs_ratio = 1.73
    # vs = [v / vp_vs_ratio for v in vp]
    # h = 0.3

    # ##################################### DEMO DATA #####################################

    ##################################### GaMMA Paper DATA #####################################
    region = "gamma_paper/"
    data_path = f"./{region}/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    picks_file = os.path.join(data_path, "picks.csv")
    events_file = os.path.join(data_path, "events.csv")
    stations_file = os.path.join(data_path, "stations.csv")

    os.system(f"[ -f {events_file} ] || curl -L https://osf.io/download/945dq/ -o {events_file}")
    os.system(f"[ -f {picks_file} ] || curl -L https://osf.io/download/gwxtn/ -o {picks_file}")
    os.system(f"[ -f {stations_file} ] || curl -L https://osf.io/download/km97w/ -o {stations_file}")

    picks = pd.read_csv(picks_file, sep="\t")
    picks.rename(
        {
            "id": "station_id",
            "timestamp": "phase_time",
            "type": "phase_type",
            "prob": "phase_score",
            "amp": "phase_amplitude",
            "event_idx": "event_index",
        },
        axis=1,
        inplace=True,
    )
    picks["phase_type"] = picks["phase_type"].str.upper()
    picks["phase_time"] = pd.to_datetime(picks["phase_time"])
    events = pd.read_csv(events_file, sep="\t")
    events.rename({"event_idx": "event_index"}, axis=1, inplace=True)
    events["depth_km"] = events["depth(m)"] / 1000.0
    events["time"] = pd.to_datetime(events["time"])

    # picks = picks[picks["phase_time"] < pd.to_datetime("2019-07-05 00:00:00")]
    # events = events[events["time"] < pd.to_datetime("2019-07-05 00:00:00")]

    stations = pd.read_csv(stations_file, sep="\t")
    stations.rename({"station": "station_id", "elevation(m)": "elevation_m"}, axis=1, inplace=True)

    picks.to_csv(os.path.join(data_path, "gamma_picks.csv"), index=False)
    events.to_csv(os.path.join(data_path, "gamma_events.csv"), index=False)
    stations.to_csv(os.path.join(data_path, "gamma_stations.csv"), index=False)

    config = {
        "minlatitude": 35.205,
        "maxlatitude": 36.205,
        "minlongitude": -118.004,
        "maxlongitude": -117.004,
        "mindepth": 0.0,
        "maxdepth": 30.0,
    }
    config["use_amplitude"] = True

    # ## Eikonal for 1D velocity model
    zz = [0.0, 5.5, 16.0, 32.0]
    vp = [5.5, 5.5, 6.7, 7.8]
    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 0.3
    # zz = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 30.0]
    # vp = [4.746, 4.793, 4.799, 5.045, 5.721, 5.879, 6.504, 6.708, 6.725, 7.800]
    # vs = [2.469, 2.470, 2.929, 2.930, 3.402, 3.403, 3.848, 3.907, 3.963, 4.500]
    # h = 0.3

    ##################################### GaMMA Paper DATA #####################################

    # ##################################### Stanford DATA #####################################
    # region = "stanford"
    # data_path = f"./{region}/"

    # picks = pd.read_csv(f"{data_path}/phase.csv", parse_dates=["time"])
    # stations = pd.read_csv(f"{data_path}/station.csv")
    # events = None

    # picks.rename({"time": "phase_time", "evid": "event_index", "phase": "phase_type"}, axis=1, inplace=True)
    # picks["phase_time"] = pd.to_datetime(picks["phase_time"])
    # picks["station_id"] = picks["network"] + "." + picks["station"]
    # picks["phase_score"] = 1.0

    # stations.rename({"elevation": "elevation_m"}, axis=1, inplace=True)
    # stations["station_id"] = stations["network"] + "." + stations["station"]

    # config = {
    #     "maxlongitude": -117.10,
    #     "minlongitude": -118.2,
    #     "maxlatitude": 36.4,
    #     "minlatitude": 35.3,
    #     "mindepth": 0.0,
    #     "maxdepth": 15.0,
    # }
    # config["use_amplitude"] = False

    # ## Eikonal for 1D velocity model
    # zz = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 30.0]
    # vp = [4.746, 4.793, 4.799, 5.045, 5.721, 5.879, 6.504, 6.708, 6.725, 7.800]
    # vs = [2.469, 2.470, 2.929, 2.930, 3.402, 3.403, 3.848, 3.907, 3.963, 4.500]
    # h = 0.3

    # ##################################### Stanford DATA #####################################

    # %%
    result_path = f"results/{region}/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    figure_path = f"figures/{region}/"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # %%
    ## Automatic region; you can also specify a region
    # lon0 = stations["longitude"].median()
    # lat0 = stations["latitude"].median()
    lon0 = (config["minlongitude"] + config["maxlongitude"]) / 2
    lat0 = (config["minlatitude"] + config["maxlatitude"]) / 2
    proj = Proj(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0}  +units=km")

    # %%
    stations["depth_km"] = -stations["elevation_m"] / 1000
    if "station_term_time_p" not in stations.columns:
        stations["station_term_time_p"] = 0.0
    if "station_term_time_s" not in stations.columns:
        stations["station_term_time_s"] = 0.0
    if "station_term_amplitude" not in stations.columns:
        stations["station_term_amplitude"] = 0.0
    stations[["x_km", "y_km"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["z_km"] = stations["elevation_m"].apply(lambda x: -x / 1e3)

    if events is not None:
        events[["x_km", "y_km"]] = events.apply(
            lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
        )
        events["z_km"] = events["depth_km"] if "depth_km" in events.columns else 10.0

    ## set up the config; you can also specify the region manually
    if ("xlim_km" not in config) or ("ylim_km" not in config) or ("zlim_km" not in config):

        # project minlatitude, maxlatitude, minlongitude, maxlongitude to ymin, ymax, xmin, xmax
        xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
        xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
        zmin, zmax = config["mindepth"], config["maxdepth"]
        config["xlim_km"] = (xmin, xmax)
        config["ylim_km"] = (ymin, ymax)
        config["zlim_km"] = (zmin, zmax)

    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}

    # %%
    config["eikonal"] = None

    # # ## Eikonal for 1D velocity model
    # zz = [0.0, 5.5, 16.0, 32.0]
    # vp = [5.5, 5.5, 6.7, 7.8]
    # vp_vs_ratio = 1.73
    # vs = [v / vp_vs_ratio for v in vp]
    # # Northern California (Gil7)
    # # zz = [0.0, 1.0, 3.0, 4.0, 5.0, 17.0, 25.0, 62.0]
    # # vp = [3.2, 3.2, 4.5, 4.8, 5.51, 6.21, 6.89, 7.83]
    # # vs = [1.5, 1.5, 2.4, 2.78, 3.18, 3.40, 3.98, 4.52]
    # h = 0.3

    vel = {"Z": zz, "P": vp, "S": vs}
    config["eikonal"] = {
        "vel": vel,
        "h": h,
        "xlim_km": config["xlim_km"],
        "ylim_km": config["ylim_km"],
        "zlim_km": config["zlim_km"],
    }
    config["eikonal"] = init_eikonal2d(config["eikonal"])

    # %% config for location
    config["min_picks"] = 6
    config["min_picks_ratio"] = 0.5
    config["max_residual_time"] = 1.0
    config["max_residual_amplitude"] = 1.0
    # config["max_residual_time"] = 0.2  ## Forge
    # config["max_residual_time"] = 0.2 ## Stanford
    # config["max_residual_amplitude"] = 0.2 ## Stanford
    # config["min_score"] = 0.6
    # config["min_p_picks"] = 1
    # config["min_s_picks"] = 1
    config["min_score"] = 0.5
    config["min_s_picks"] = 1.5
    config["min_p_picks"] = 1.5

    config["bfgs_bounds"] = (
        (config["xlim_km"][0] - 1, config["xlim_km"][1] + 1),  # x
        (config["ylim_km"][0] - 1, config["ylim_km"][1] + 1),  # y
        # (config["zlim_km"][0], config["zlim_km"][1] + 1),  # z
        (0, config["zlim_km"][1] + 1),
        (None, None),  # t
    )

    # %%
    mapping_phase_type_int = {"P": 0, "S": 1}
    config["vel"] = {mapping_phase_type_int[k]: v for k, v in config["vel"].items()}
    picks["phase_type"] = picks["phase_type"].map(mapping_phase_type_int)
    if "phase_amplitude" in picks.columns:
        picks["phase_amplitude"] = picks["phase_amplitude"].apply(lambda x: np.log10(x) + 2.0)  # convert to log10(cm/s)

    # %%
    # reindex in case the index does not start from 0 or is not continuous
    stations["idx_sta"] = np.arange(len(stations))
    if events is not None:
        # reindex in case the index does not start from 0 or is not continuous
        events["idx_eve"] = np.arange(len(events))

    else:
        picks = picks.merge(stations[["station_id", "x_km", "y_km", "z_km"]], on="station_id")
        events = picks.groupby("event_index").agg({"x_km": "mean", "y_km": "mean", "z_km": "mean", "phase_time": "min"})
        picks.drop(["x_km", "y_km", "z_km"], axis=1, inplace=True)
        events["z_km"] = 10.0  # km default depth
        events.rename({"phase_time": "time"}, axis=1, inplace=True)
        events["event_index"] = events.index
        events.reset_index(drop=True, inplace=True)
        events["idx_eve"] = np.arange(len(events))

    picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    print(f"Number of picks: {len(picks)}")
    print(f"Number of events: {len(events)}")

    # %%
    estimator = ADLoc(config, stations=stations[["x_km", "y_km", "z_km"]].values, eikonal=config["eikonal"])

    # %%
    NCPU = mp.cpu_count()
    MAX_SST_ITER = 10
    # MIN_SST_S = 0.01
    events_init = events.copy()

    for iter in range(MAX_SST_ITER):
        # picks, events = invert_location_iter(picks, stations, config, estimator, events_init=events_init, iter=iter)
        picks, events = invert_location(picks, stations, config, estimator, events_init=events_init, iter=iter)

        station_term_amp = (
            picks[picks["mask"] == 1.0].groupby("idx_sta").agg({"residual_amplitude": "median"}).reset_index()
        )
        station_term_amp.set_index("idx_sta", inplace=True)
        stations["station_term_amplitude"] += stations["idx_sta"].map(station_term_amp["residual_amplitude"]).fillna(0)

        ## Same P and S station term
        # station_term_time = picks[picks["mask"] == 1.0].groupby("idx_sta").agg({"residual_time": "mean"}).reset_index()
        # stations["station_term_time_p"] += (
        #     stations["idx_sta"].map(station_term_time.set_index("idx_sta")["residual_time"]).fillna(0)
        # )
        # stations["station_term_time_s"] += (
        #     stations["idx_sta"].map(station_term_time.set_index("idx_sta")["residual_time"]).fillna(0)
        # )

        ## Separate P and S station term
        station_term_time = (
            picks[picks["mask"] == 1.0].groupby(["idx_sta", "phase_type"]).agg({"residual_time": "mean"}).reset_index()
        )
        station_term_time.set_index("idx_sta", inplace=True)
        stations["station_term_time_p"] += (
            stations["idx_sta"].map(station_term_time[station_term_time["phase_type"] == 0]["residual_time"]).fillna(0)
        )
        stations["station_term_time_s"] += (
            stations["idx_sta"].map(station_term_time[station_term_time["phase_type"] == 1]["residual_time"]).fillna(0)
        )

        plotting_ransac(stations, figure_path, config, picks, events_init, events, suffix=f"_adloc_sst_{iter}")

        if "event_index" not in events.columns:
            events["event_index"] = events.merge(picks[["idx_eve", "event_index"]], on="idx_eve")["event_index"]
        events[["longitude", "latitude"]] = events.apply(
            lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
        )
        events["depth_km"] = events["z_km"]

        picks["adloc_mask"] = picks["mask"]
        picks["adloc_residual_time"] = picks["residual_time"]
        picks["adloc_residual_amplitude"] = picks["residual_amplitude"]
        picks.to_csv(os.path.join(result_path, f"adloc_picks_sst_{iter}.csv"), index=False)
        events.to_csv(os.path.join(result_path, f"adloc_events_sst_{iter}.csv"), index=False)
        stations.to_csv(os.path.join(result_path, f"adloc_stations_sst_{iter}.csv"), index=False)

        if iter == 0:
            MIN_SST_S = (
                np.mean(np.abs(station_term_time["residual_time"])) / 10.0
            )  # break at 10% of the initial station term
            print(f"MIN_SST (s): {MIN_SST_S}")
        if np.mean(np.abs(station_term_time["residual_time"])) < MIN_SST_S:
            print(f"Mean station term: {np.mean(np.abs(station_term_time['residual_time']))}")
            # break
        iter += 1

    # %%
    plotting_ransac(stations, figure_path, config, picks, events_init, events, suffix=f"_ransac")

    if "event_index" not in events.columns:
        events["event_index"] = events.merge(picks[["idx_eve", "event_index"]], on="idx_eve")["event_index"]
    events[["longitude", "latitude"]] = events.apply(
        lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
    )
    events["depth_km"] = events["z_km"]
    events.drop(["idx_eve", "x_km", "y_km", "z_km"], axis=1, inplace=True, errors="ignore")
    events.sort_values(["time"], inplace=True)

    # picks.rename({"mask": "adloc_mask", "residual": "adloc_residual"}, axis=1, inplace=True)
    picks["phase_type"] = picks["phase_type"].map({0: "P", 1: "S"})
    picks.drop(
        ["idx_eve", "idx_sta", "mask", "residual_time", "residual_amplitude"], axis=1, inplace=True, errors="ignore"
    )
    picks.sort_values(["phase_time"], inplace=True)

    stations.drop(["idx_sta", "x_km", "y_km", "z_km"], axis=1, inplace=True, errors="ignore")
    # stations.rename({"station_term": "adloc_station_term_s"}, axis=1, inplace=True)

    picks.to_csv(os.path.join(result_path, "adloc_picks.csv"), index=False)
    events.to_csv(os.path.join(result_path, "adloc_events.csv"), index=False)
    stations.to_csv(os.path.join(result_path, "adloc_stations.csv"), index=False)

# %%
