# %%
import matplotlib
import numpy as np
import scipy
from eikonal2d import eikonal_solve, grad_traveltime, traveltime
from sklearn.base import BaseEstimator
from sklearn.linear_model import RANSACRegressor

np.random.seed(0)
# matplotlib.use("Agg")


class ADLoc(BaseEstimator):
    def __init__(self, config, stations, num_event=3, events=None, eikonal=None):
        """
        events: [x, y, z, t]
        """
        xlim = config["xlim_km"]
        ylim = config["ylim_km"]
        zlim = config["zlim_km"]
        vel = config["vel"]
        self.config = config
        self.stations = stations

        self.vel = vel
        self.num_event = num_event

        self.eikonal = eikonal

        if events is not None:
            assert events.shape == (num_event, 4)
            self.events = events
        else:
            self.events = np.array(
                [[(xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2, (zlim[0] + zlim[1]) / 2, 0]] * num_event
            )

    @staticmethod
    def loss_grad(event, X, y, vel={0: 6.0, 1: 6.0 / 1.75}, stations=None, eikonal=None):
        """
        X: data_frame with columns ["timestamp", "x_km", "y_km", "z_km", "type"]
        """
        station_index = X[:, 0]
        phase_type = X[:, 1]

        if eikonal is None:
            v = np.array([vel[t] for t in phase_type])
            tt = np.linalg.norm(event[:3] - stations[station_index, :3], axis=-1) / v + event[3]
        else:
            tt = traveltime(0, station_index, phase_type, event[np.newaxis, :3], stations, eikonal) + event[3]
        loss = 0.5 * np.sum((tt - y) ** 2)

        J = np.ones((len(X), 4))
        if eikonal is None:
            J[:, :3] = (
                (event[:3] - stations[station_index, :3])
                / np.linalg.norm(event[:3] - stations[station_index, :3], axis=-1, keepdims=True)
                / v[:, np.newaxis]
            )
        else:
            grad = grad_traveltime(0, station_index, phase_type, event[np.newaxis, :3], stations, eikonal)
            J[:, :3] = grad

        J = np.sum((tt - y)[:, np.newaxis] * J, axis=0)
        return loss, J

    def fit(self, X, y=None, event_index=0):

        opt = scipy.optimize.minimize(
            self.loss_grad,
            x0=self.events[event_index],
            method="L-BFGS-B",
            jac=True,
            args=(X, y, self.vel, self.stations, self.eikonal),
            bounds=[
                (self.config["xlim_km"][0], self.config["xlim_km"][1]),
                (self.config["ylim_km"][0], self.config["ylim_km"][1]),
                (self.config["zlim_km"][0], self.config["zlim_km"][1]),
                (None, None),
            ],
        )

        self.events[event_index, :] = opt.x
        self.is_fitted_ = True

        return self

    def predict(self, X, event_index=0):
        """
        X: data_frame with columns ["timestamp", "x_km", "y_km", "z_km", "type"]
        """
        station_index = X[:, 0]
        phase_type = X[:, 1]

        if self.eikonal is None:
            v = np.array([self.vel[t] for t in phase_type])
            tt = (
                np.linalg.norm(self.events[event_index, :3] - self.stations[station_index, :3], axis=-1) / v
                + self.events[event_index, 3]
            )
        else:
            tt = (
                traveltime(event_index, station_index, phase_type, self.events, self.stations, self.eikonal)
                + self.events[event_index, 3]
            )

        return tt

    def score(self, X, y=None, event_index=0):
        """
        X: data_frame with columns ["timestamp", "x_km", "y_km", "z_km", "type"]
        """
        if len(X) == 0:
            return 0
        tt = self.predict(X, event_index)
        R2 = 1 - np.sum((y - tt) ** 2) / np.sum((y - np.mean(y)) ** 2)
        print(f"{R2=}")
        return R2


def init_eikonal2d(config):

    # nr = 21
    # nz = 21
    # vel = {"p": 6.0, "s": 6.0 / 1.73}

    xlim = config["xlim_km"]
    ylim = config["ylim_km"]
    zlim = config["zlim_km"]
    # vel = config["vel"]
    if "vel" in config:
        vel = config["vel"]
    else:
        vel = {"p": 6.0, "s": 6.0 / 1.73}
    if "h" in config:
        h = config["h"]
    else:
        h = 1.0
    nr = int(np.sqrt((xlim[1] - xlim[0]) ** 2 + (ylim[1] - ylim[0]) ** 2) / h)
    nz = int((zlim[1] - zlim[0]) / h)
    rgrid = np.arange(nr) * h
    zgrid = np.arange(nz) * h

    print(f"{nr=}, {nz=}")
    print(f"{len(stations) = }")
    vp = np.ones((nr, nz)) * vel["p"]
    vs = np.ones((nr, nz)) * vel["s"]

    up = 1000 * np.ones((nr, nz))
    up[0, 0] = 0.0

    up = eikonal_solve(up, vp, h)
    grad_up = np.gradient(up, h, edge_order=2)
    up = up.ravel()
    grad_up = [x.ravel() for x in grad_up]

    us = 1000 * np.ones((nr, nz))
    us[0, 0] = 0.1

    us = eikonal_solve(us, vs, h)
    grad_us = np.gradient(us, h, edge_order=2)
    us = us.ravel()
    grad_us = [x.ravel() for x in grad_us]

    config = {
        "up": up,
        "us": us,
        "grad_up": grad_up,
        "grad_us": grad_us,
        "rgrid": rgrid,
        "zgrid": zgrid,
        "nr": nr,
        "nz": nz,
        "h": h,
    }

    return config


# %%
if __name__ == "__main__":

    # %%
    import json
    import os

    import matplotlib.pyplot as plt
    import pandas as pd

    # data_path = "../tests/results"
    data_path = "./results"
    events = pd.read_csv(os.path.join(data_path, "events.csv"), parse_dates=["time"])
    stations = pd.read_csv(os.path.join(data_path, "stations.csv"))
    picks = pd.read_csv(os.path.join(data_path, "picks.csv"), parse_dates=["phase_time"])
    with open(os.path.join(data_path, "config.json"), "r") as f:
        config = json.load(f)

    eikonal = init_eikonal2d(config)
    eikonal = None

    # %%
    stations["idx_sta"] = stations.index  # reindex in case the index does not start from 0 or is not continuous

    events["idx_eve"] = events.index  # reindex in case the index does not start from 0 or is not continuous

    picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")

    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    # # %%
    # for idx_sta, picks_station in picks.groupby("idx_sta"):
    #     station_loc = stations.loc[idx_sta]
    #     print(f"Station {station_loc['station_id']} at ({station_loc['x_km']}, {station_loc['y_km']})")
    #     for _, pick in picks_station.iterrows():
    #         event_loc = events.loc[pick["idx_eve"]]
    #         print(f"Event {event_loc['event_index']} at ({event_loc['x_km']}, {event_loc['y_km']})")
    #     raise

    # %%
    for idx_eve, picks_event in picks.groupby("idx_eve"):
        event_loc = events.loc[idx_eve]
        print(f"Event {event_loc['event_index']} at ({event_loc['x_km']}, {event_loc['y_km']})")
        for _, pick in picks_event.iterrows():
            station_loc = stations.loc[pick["idx_sta"]]
            print(f"Station {station_loc['station_id']} at ({station_loc['x_km']}, {station_loc['y_km']})")
        break

    # %%
    plt.figure()
    # plt.scatter(picks_event["x_km"], picks_event["y_km"], c=picks_event["phase_time"])
    tmp = picks_event.merge(stations[["x_km", "y_km", "z_km", "station_id"]], on="station_id")
    tmp["dist_km"] = tmp[["x_km", "y_km", "z_km"]].apply(
        lambda x: np.linalg.norm(x - event_loc[["x_km", "y_km", "z_km"]]), axis=1
    )
    plt.scatter(tmp["x_km"], tmp["y_km"], c=tmp["phase_time"], cmap="viridis")
    plt.scatter(event_loc["x_km"], event_loc["y_km"], c="r")
    plt.colorbar()
    plt.show()

    plt.figure()
    colors = lambda x: "r" if x == "P" else "b"
    plt.scatter(tmp["phase_time"], tmp["dist_km"], c=tmp["phase_type"].apply(colors))
    plt.show()

    # %%

    # %%
    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}
    X = picks_event.merge(
        stations[["x_km", "y_km", "z_km", "station_id"]],
        on="station_id",
    )
    t0 = X["phase_time"].min()
    X.rename(columns={"phase_type": "type", "phase_time": "t_s"}, inplace=True)
    X["t_s"] = (X["t_s"] - t0).dt.total_seconds()
    X = X[["x_km", "y_km", "z_km", "t_s", "type", "idx_sta"]]
    mapping_int = {"P": 0, "S": 1}
    config["vel"] = {mapping_int[k]: v for k, v in config["vel"].items()}
    X["type"] = X["type"].apply(lambda x: mapping_int[x.upper()])

    # estimator = ADLoc(config_, event_loc=np.array([x0, y0, z0]), event_t=(origin_time - t0).total_seconds())
    estimator = ADLoc(config, stations=stations[["x_km", "y_km", "z_km"]].values, eikonal=eikonal)
    # tt = estimator.predict(X[["x_km", "y_km", "z_km", "type"]].values)
    # estimator.score(X[["x_km", "y_km", "z_km", "type"]].values, y=X["t_s"].values)
    tt = estimator.predict(X[["idx_sta", "type"]].values)
    estimator.score(X[["idx_sta", "type"]].values, y=X["t_s"].values)

    print(f"True event loc: {event_loc[['x_km', 'y_km', 'z_km']].values}")
    print(f"Init event loc: {estimator.events[:3]}")

    # %%
    plt.figure()
    event_index = 0
    X["dist_km"] = X[["x_km", "y_km", "z_km"]].apply(
        lambda x: np.linalg.norm(x - estimator.events[event_index, :3]), axis=1
    )
    colors = lambda x: "r" if x == 0 else "b"
    plt.scatter(X["t_s"], X["dist_km"], c=X["type"].apply(colors), marker="x")
    plt.scatter(tt, X["dist_km"], c=X["type"].apply(colors), marker="o")
    plt.show()

    # %%
    # estimator.fit(X[["x_km", "y_km", "z_km", "type"]].values, y=X["t_s"].values)
    # estimator.score(X[["x_km", "y_km", "z_km", "type"]].values, y=X["t_s"].values)
    # tt = estimator.predict(X[["x_km", "y_km", "z_km", "type"]].values)
    estimator.fit(X[["idx_sta", "type"]].values, y=X["t_s"].values)
    estimator.score(X[["idx_sta", "type"]].values, y=X["t_s"].values)
    tt = estimator.predict(X[["idx_sta", "type"]].values)
    print(f"True event loc: {event_loc[['x_km', 'y_km', 'z_km']].values}")
    print(f"Estimated event loc: {estimator.events[:3]}")

    # %%
    plt.figure()
    X["dist_km"] = X[["x_km", "y_km", "z_km"]].apply(
        lambda x: np.linalg.norm(x - estimator.events[event_index, :3]), axis=1
    )
    colors = lambda x: "r" if x == 0 else "b"
    plt.scatter(X["t_s"], X["dist_km"], c=X["type"].apply(colors), marker="x")
    plt.scatter(tt, X["dist_km"], c=X["type"].apply(colors), marker="o")
    plt.show()

    # %%
    # reg = RANSACRegressor(estimator=ADLoc(config_, event_loc=np.array([[x0, y0, 0]]), event_t=0), random_state=0, min_samples=10, residual_threshold=4.0).fit(X[["x_km", "y_km", "z_km", "vel"]].values, X["t_s"].values)
    reg = RANSACRegressor(
        estimator=ADLoc(config, stations=stations[["x_km", "y_km", "z_km"]].values, eikonal=eikonal),
        random_state=0,
        min_samples=4,
        residual_threshold=1.0,
    )
    # reg.fit(X[["x_km", "y_km", "z_km", "type"]].values, X["t_s"].values)
    reg.fit(X[["idx_sta", "type"]].values, X["t_s"].values)
    # reg = RANSACRegressor(estimator=ADLoc(config_), random_state=0, min_samples=4, residual_threshold=1.0).fit(X[["x_km", "y_km", "z_km", "type"]].values, X["t_s"].values)
    mask = reg.inlier_mask_
    estimator = reg.estimator_
    # estimator.score(X[["x_km", "y_km", "z_km", "type"]].values, y=X["t_s"].values)
    # tt = estimator.predict(X[["x_km", "y_km", "z_km", "type"]].values)
    estimator.score(X[["idx_sta", "type"]].values, y=X["t_s"].values)
    tt = estimator.predict(X[["idx_sta", "type"]].values)
    print(f"True event loc: {event_loc[['x_km', 'y_km', 'z_km']].values}")
    print(f"Estimated event loc: {estimator.events[:3]}")

    # %%
    plt.figure()
    X["dist_km"] = X[["x_km", "y_km", "z_km"]].apply(
        lambda x: np.linalg.norm(x - estimator.events[event_index, :3]), axis=1
    )
    colors = lambda x: "r" if x == 0 else "b"
    plt.scatter(X["t_s"], X["dist_km"], c=X["type"].apply(colors), marker="x")
    plt.scatter(tt, X["dist_km"], c=X["type"].apply(colors), marker="o")
    plt.show()

    # %% add random picks
    # %%
    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}
    X = picks_event.merge(stations[["x_km", "y_km", "z_km", "station_id"]], on="station_id")
    t0 = X["phase_time"].min()
    X.rename(columns={"phase_type": "type", "phase_time": "t_s"}, inplace=True)
    X["t_s"] = (X["t_s"] - t0).dt.total_seconds()
    X = X[["x_km", "y_km", "z_km", "t_s", "type", "idx_sta"]]
    mapping_int = {"P": 0, "S": 1}
    config["vel"] = {mapping_int[k]: v for k, v in config["vel"].items()}
    X["type"] = X["type"].apply(lambda x: mapping_int[x.upper()])

    num_noise = 10
    noise = pd.DataFrame(
        {
            "x_km": np.random.rand(num_noise) * (X["x_km"].max() - X["x_km"].min()) + X["x_km"].min(),
            "y_km": np.random.rand(num_noise) * (X["y_km"].max() - X["y_km"].min()) + X["y_km"].min(),
            "z_km": np.random.rand(num_noise) * (X["z_km"].max() - X["z_km"].min()) + X["z_km"].min(),
            "t_s": np.random.rand(num_noise) * (X["t_s"].max() - X["t_s"].min()) + X["t_s"].min(),
            "type": np.random.choice([0, 1], num_noise),
            "idx_sta": np.random.choice(X["idx_sta"], num_noise),
            "mask": [0] * num_noise,
        }
    )
    X["mask"] = 1
    X = pd.concat([X, noise], ignore_index=True)
    X["dist_km"] = X[["x_km", "y_km", "z_km"]].apply(
        lambda x: np.linalg.norm(x - event_loc[["x_km", "y_km", "z_km"]]), axis=1
    )
    plt.figure()
    colors = lambda x: "r" if x == 0 else "b"
    plt.scatter(X["t_s"], X["dist_km"], c=X["type"].apply(colors), marker="o")
    plt.scatter(X["t_s"][X["mask"] == 0], X["dist_km"][X["mask"] == 0], c="k", marker="o", alpha=0.6)
    plt.show()

    # %%

    # reg = RANSACRegressor(estimator=ADLoc(config_, event_loc=np.array([[x0, y0, 0]]), event_t=0), random_state=0, min_samples=10, residual_threshold=4.0).fit(X[["x_km", "y_km", "z_km", "vel"]].values, X["t_s"].values)
    reg = RANSACRegressor(
        estimator=ADLoc(config, stations=stations[["x_km", "y_km", "z_km"]].values, eikonal=eikonal),
        random_state=0,
        min_samples=4,
        residual_threshold=1.0,
    )
    # reg.fit(X[["x_km", "y_km", "z_km", "type"]].values, X["t_s"].values)
    reg.fit(X[["idx_sta", "type"]].values, X["t_s"].values)
    # reg = RANSACRegressor(estimator=ADLoc(config_), random_state=0, min_samples=4, residual_threshold=1.0).fit(X[["x_km", "y_km", "z_km", "type"]].values, X["t_s"].values)
    mask = reg.inlier_mask_
    estimator = reg.estimator_
    # estimator.score(X[["x_km", "y_km", "z_km", "type"]].values, y=X["t_s"].values)
    # tt = estimator.predict(X[["x_km", "y_km", "z_km", "type"]].values)
    estimator.score(X[["idx_sta", "type"]].values, y=X["t_s"].values)
    tt = estimator.predict(X[["idx_sta", "type"]].values)
    print(f"True event loc: {event_loc[['x_km', 'y_km', 'z_km']].values}")
    print(f"Estimated event loc: {estimator.events[:3]}")
    # %%
    plt.figure()
    X["dist_km"] = X[["x_km", "y_km", "z_km"]].apply(
        lambda x: np.linalg.norm(x - estimator.events[event_index, :3]), axis=1
    )
    colors = lambda x: "r" if x == 0 else "b"
    plt.scatter(X["t_s"], X["dist_km"], c=X["type"].apply(colors), marker="o")
    plt.scatter(tt, X["dist_km"], c=X["type"].apply(colors), marker="x", s=100)
    plt.scatter(X["t_s"][~mask], X["dist_km"][~mask], c="k", marker="o", alpha=0.6)
    plt.show()

# %%
