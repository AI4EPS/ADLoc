# %%
import json
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
from matplotlib import pyplot as plt
from pyproj import Proj
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from adloc.adloc import TravelTimeDD
from adloc.data import PhaseDatasetDD
from adloc.eikonal2d import init_eikonal2d
from adloc.inversion import optimize_dd
from utils import plotting_dd

torch.manual_seed(0)
np.random.seed(0)


# %%
if __name__ == "__main__":

    # %%
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend="gloo")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        print(f"DDP rank {ddp_rank}, local rank {ddp_local_rank}, world size {ddp_world_size}")
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        print("Non-DDP run")

    # %%
    # region = "synthetic"
    region = "ridgecrest"
    data_path = f"test_data/{region}"
    result_path = f"results/{region}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    figure_path = f"figures/{region}/"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    picks_file = os.path.join(data_path, "gamma_picks.csv")
    events_file = os.path.join(data_path, "gamma_events.csv")
    stations_file = os.path.join(data_path, "stations.csv")

    # %% generate the double-difference pair file
    if ddp_local_rank == 0:
        if (not os.path.exists(os.path.join(result_path, "adloc_dt.dat"))) or (
            input("Regenerate the double-difference pair file (adloc_dt.dat)? (N/y): ") == "y"
        ):
            os.system(
                f"python generate_pairs.py --stations {stations_file} --events {events_file} --picks {picks_file} --result_path {result_path}"
            )

    if ddp:
        dist.barrier()

    # %% reading from the generated files
    config = json.load(open(os.path.join(data_path, "config.json")))
    events = pd.read_csv(os.path.join(result_path, "pair_events.csv"), parse_dates=["time"])
    stations = pd.read_csv(os.path.join(result_path, "pair_stations.csv"))
    picks = pd.read_csv(os.path.join(result_path, "pair_picks.csv"), parse_dates=["phase_time"])
    dtypes = pickle.load(open(os.path.join(result_path, "pair_dtypes.pkl"), "rb"))
    pairs = np.memmap(os.path.join(result_path, "pair_dt.dat"), mode="r", dtype=dtypes)
    events_init = events.copy()

    # %%
    ## Automatic region; you can also specify a region
    lon0 = stations["longitude"].median()
    lat0 = stations["latitude"].median()
    proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")

    ## set up the config; you can also specify the region manually
    if ("xlim_km" not in config) or ("ylim_km" not in config) or ("zlim_km" not in config):
        xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
        xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
        zmin = stations["z_km"].min()
        zmax = 20
        config = {}
        config["xlim_km"] = (xmin, xmax)
        config["ylim_km"] = (ymin, ymax)
        config["zlim_km"] = (zmin, zmax)

    mapping_phase_type_int = {"P": 0, "S": 1}
    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}
    config["vel"] = {mapping_phase_type_int[k]: v for k, v in config["vel"].items()}

    # %%
    config["eikonal"] = None
    ## Eikonal for 1D velocity model
    zz = [0.0, 5.5, 16.0, 32.0]
    vp = [5.5, 5.5, 6.7, 7.8]
    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 0.3
    vel = {"Z": zz, "P": vp, "S": vs}
    config["eikonal"] = {
        "vel": vel,
        "h": h,
        "xlim_km": config["xlim_km"],
        "ylim_km": config["ylim_km"],
        "zlim_km": config["zlim_km"],
    }
    config["eikonal"] = init_eikonal2d(config["eikonal"])

    # %%
    config["bfgs_bounds"] = (
        (config["xlim_km"][0] - 1, config["xlim_km"][1] + 1),  # x
        (config["ylim_km"][0] - 1, config["ylim_km"][1] + 1),  # y
        (0, config["zlim_km"][1] + 1),
        (None, None),  # t
    )

    # %%
    if ddp_local_rank == 0:
        plotting_dd(events, stations, config, figure_path, events_init, iter=0)

    # %%
    phase_dataset = PhaseDatasetDD(pairs, picks, events, stations, rank=ddp_local_rank, world_size=ddp_world_size)
    data_loader = DataLoader(phase_dataset, batch_size=None, shuffle=False, num_workers=0)

    # %%
    num_event = len(events)
    num_station = len(stations)
    station_loc = stations[["x_km", "y_km", "z_km"]].values
    # event_loc_init = np.zeros((num_event, 3))
    # event_loc_init[:, 2] = np.mean(config["zlim_km"])
    event_loc_init = events[["x_km", "y_km", "z_km"]].values  # + np.random.randn(num_event, 3) * 2.0
    travel_time = TravelTimeDD(
        num_event,
        num_station,
        station_loc,
        event_loc=event_loc_init,  # Initial location
        # event_time=event_time,
        eikonal=config["eikonal"],
    )
    if ddp:
        travel_time = DDP(travel_time)
    raw_travel_time = travel_time.module if ddp else travel_time

    if ddp_local_rank == 0:
        print(
            f"Dataset: {len(picks)} picks, {len(events)} events, {len(stations)} stations, {len(data_loader)} batches"
        )

    ## invert loss
    ######################################################################################################
    optimizer = optim.Adam(params=travel_time.parameters(), lr=0.1)
    EPOCHS = 100
    for i in range(EPOCHS):
        loss = 0
        optimizer.zero_grad()
        # for meta in tqdm(phase_dataset, desc=f"Epoch {i}"):
        if ddp_local_rank == 0:
            print(f"Epoch {i}: ", end="")
        for meta in data_loader:
            if ddp_local_rank == 0:
                print(".", end="")

            loss_ = travel_time(
                meta["idx_sta"],
                meta["idx_eve"],
                meta["phase_type"],
                meta["phase_time"],
                meta["phase_weight"],
            )["loss"]

            loss_.backward()

            if ddp:
                dist.all_reduce(loss_, op=dist.ReduceOp.SUM)
                # loss_ /= ddp_world_size

            loss += loss_

        optimizer.step()
        raw_travel_time.event_loc.weight.data[:, 2].clamp_(min=config["zlim_km"][0], max=config["zlim_km"][1])
        if ddp_local_rank == 0:
            print(f"Loss: {loss}")

        invert_event_loc = raw_travel_time.event_loc.weight.clone().detach().numpy()
        invert_event_time = raw_travel_time.event_time.weight.clone().detach().numpy()

        # events["time"] = events["time"] + pd.to_timedelta(np.squeeze(invert_event_time), unit="s")
        events["x_km"] = invert_event_loc[:, 0]
        events["y_km"] = invert_event_loc[:, 1]
        events["z_km"] = invert_event_loc[:, 2]
        # events[["longitude", "latitude"]] = events.apply(
        #     lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
        # )
        # events["depth_km"] = events["z_km"]
        # events.to_csv(
        #     f"{result_path}/adloc_dd_events.csv", index=False, float_format="%.5f", date_format="%Y-%m-%dT%H:%M:%S.%f"
        # )

        if ddp_local_rank == 0 and (i % 10 == 9):
            plotting_dd(events, stations, config, figure_path, events_init, iter=i + 1)

    # ######################################################################################################
    # optimizer = optim.LBFGS(params=raw_travel_time.parameters(), max_iter=10, line_search_fn="strong_wolfe")

    # def closure():
    #     optimizer.zero_grad()
    #     loss = 0
    #     # for meta in tqdm(phase_dataset, desc=f"BFGS"):
    #     if ddp_local_rank == 0:
    #         print(f"BFGS: ", end="")
    #     for meta in phase_dataset:
    #         if ddp_local_rank == 0:
    #             print(".", end="")

    #         loss_ = travel_time(
    #              meta["idx_sta"],
    #             meta["idx_eve"],
    #             meta["phase_type"],
    #             meta["phase_time"],
    #             meta["phase_weight"],
    #         )["loss"]
    #         loss_.backward()

    #         if ddp:
    #             dist.all_reduce(loss_, op=dist.ReduceOp.SUM)
    #             # loss_ /= ddp_world_size

    #         loss += loss_

    #     if ddp_local_rank == 0:
    #         print(f"Loss: {loss}")
    #     raw_travel_time.event_loc.weight.data[:, 2].clamp_(min=config["zlim_km"][0], max=config["zlim_km"][1])
    #     return loss

    # optimizer.step(closure)
    # ######################################################################################################

    # %%
    if ddp_local_rank == 0:
        invert_event_loc = raw_travel_time.event_loc.weight.clone().detach().numpy()
        invert_event_time = raw_travel_time.event_time.weight.clone().detach().numpy()

        events["time"] = events["time"] + pd.to_timedelta(np.squeeze(invert_event_time), unit="s")
        events["x_km"] = invert_event_loc[:, 0]
        events["y_km"] = invert_event_loc[:, 1]
        events["z_km"] = invert_event_loc[:, 2]
        events[["longitude", "latitude"]] = events.apply(
            lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
        )
        events["depth_km"] = events["z_km"]
        events.to_csv(
            f"{result_path}/adloc_dd_events.csv", index=False, float_format="%.5f", date_format="%Y-%m-%dT%H:%M:%S.%f"
        )
