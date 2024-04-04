import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from tqdm import tqdm

# class PhaseDataset(Dataset):
#     def __init__(self, picks, events, stations, batch_size=100, double_difference=False, config=None):
#         self.picks = picks
#         self.events = events
#         self.stations = stations
#         self.config = config
#         self.double_difference = double_difference
#         self.batch_size = batch_size
#         if double_difference:
#             self.read_data_dd()
#         else:
#             self.read_data()

#     def __len__(self):
#         ## TODO: return batch
#         return 1
#         # return (len(self.events) - 1) // self.batch_size + 1

#     def read_data(self):
#         event_index = []
#         station_index = []
#         phase_score = []
#         phase_time = []
#         phase_type = []

#         picks_by_event = self.picks.groupby("index")
#         for key, group in picks_by_event:
#             if key == -1:
#                 continue
#             phase_time.append(group["phase_time"].values)
#             phase_score.append(group["phase_score"].values)
#             phase_type.extend(group["phase_type"].values.tolist())
#             event_index.extend([key] * len(group))
#             station_index.append(self.stations.loc[group["station_id"], "index"].values)

#         phase_time = np.concatenate(phase_time)
#         phase_score = np.concatenate(phase_score)
#         phase_type = np.array([{"P": 0, "S": 1}[x.upper()] for x in phase_type])
#         event_index = np.array(event_index)
#         station_index = np.concatenate(station_index)

#         self.station_index = torch.tensor(station_index, dtype=torch.long)
#         self.event_index = torch.tensor(event_index, dtype=torch.long)
#         self.phase_weight = torch.tensor(phase_score, dtype=torch.float32)
#         self.phase_time = torch.tensor(phase_time, dtype=torch.float32)
#         self.phase_type = torch.tensor(phase_type, dtype=torch.long)

#     def read_data_dd(self):
#         event_index = []
#         station_index = []
#         phase_score = []
#         phase_time = []
#         phase_type = []

#         event_loc = self.events[["x_km", "y_km", "z_km"]].values
#         event_time = self.events["time"].values[:, np.newaxis]

#         neigh = NearestNeighbors(radius=self.config.min_pair_dist, metric="euclidean")
#         neigh.fit(event_loc)

#         picks_by_event = self.picks.groupby("index")

#         for key1, group1 in tqdm(picks_by_event, total=len(picks_by_event), desc="Generating pairs"):
#             if key1 == -1:
#                 continue

#             for key2 in neigh.radius_neighbors([event_loc[key1]], return_distance=False)[0]:
#                 if key1 >= key2:
#                     continue

#                 common = group1.merge(picks_by_event.get_group(key2), on=["station_id", "phase_type"], how="inner")
#                 phase_time.append(common["phase_time_x"].values - common["phase_time_y"].values)
#                 phase_score.append(common["phase_score_x"].values * common["phase_score_y"].values)
#                 phase_type.extend(common["phase_type"].values.tolist())
#                 event_index.extend([[key1, key2]] * len(common))
#                 station_index.append(self.stations.loc[common["station_id"], "index"].values)

#         if len(phase_time) > 0:
#             phase_time = np.concatenate(phase_time)
#             phase_score = np.concatenate(phase_score)
#             phase_type = np.array([{"P": 0, "S": 1}[x.upper()] for x in phase_type])
#             event_index = np.array(event_index)
#             station_index = np.concatenate(station_index)

#             # %%
#             self.station_index = torch.tensor(station_index, dtype=torch.long)
#             self.event_index = torch.tensor(event_index, dtype=torch.long)
#             self.phase_weight = torch.tensor(phase_score, dtype=torch.float32)
#             self.phase_time = torch.tensor(phase_time, dtype=torch.float32)
#             self.phase_type = torch.tensor(phase_type, dtype=torch.long)
#         else:
#             self.station_index = torch.tensor([], dtype=torch.long)
#             self.event_index = torch.tensor([], dtype=torch.long)
#             self.phase_weight = torch.tensor([], dtype=torch.float32)
#             self.phase_time = torch.tensor([], dtype=torch.float32)
#             self.phase_type = torch.tensor([], dtype=torch.long)

#     def __getitem__(self, i):
#         # phase_time = self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["phase_time"].values
#         # phase_score = self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["phase_score"].values
#         # phase_type = self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]][
#         #     "phase_type"
#         # ].values.tolist()
#         # event_index = np.array([i] * len(self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]))
#         # station_index = self.stations.loc[
#         #     self.picks[self.picks["event_index"] == self.events.loc[i, "event_index"]]["station_id"], "index"
#         # ].values

#         return {
#             "event_index": self.event_index,
#             "station_index": self.station_index,
#             "phase_time": self.phase_time,
#             "phase_weight": self.phase_weight,
#             "phase_type": self.phase_type,
#         }


class PhaseDataset(Dataset):
    def __init__(
        self,
        picks,
        events,
        stations,
        batch_size=1000,
        double_difference=False,
        min_pair_dist=10,
        max_pair_num=500,
        config=None,
    ):
        self.picks = picks
        self.events = events
        self.stations = stations
        self.config = config
        self.double_difference = double_difference
        self.batch_size = batch_size
        self.min_pair_dist = min_pair_dist
        self.max_pair_num = max_pair_num

        # preprocess
        self.picks_by_event = picks.groupby("index")
        self.event_index_batch = np.array_split(self.events.index.values, (len(self.events) - 1) // self.batch_size + 1)
        if double_difference:
            self.neigh = NearestNeighbors(radius=self.min_pair_dist)
            self.neigh.fit(self.events[["x_km", "y_km", "z_km"]].values)
            self.read_data_dd()
        else:
            self.read_data()

    def __len__(self):
        return len(self.event_index_batch)

    def read_data(self):
        meta = {}
        for i, index_batch in enumerate(self.event_index_batch):
            event_index = []
            station_index = []
            phase_score = []
            phase_time = []
            phase_type = []

            for key in index_batch:
                if key == -1:
                    continue
                group = self.picks_by_event.get_group(key)
                phase_time.append(group["phase_time"].values)
                phase_score.append(group["phase_score"].values)
                phase_type.extend(group["phase_type"].values.tolist())
                event_index.extend([key] * len(group))
                station_index.append(self.stations.loc[group["station_id"], "index"].values)

            phase_time = np.concatenate(phase_time)
            phase_score = np.concatenate(phase_score)
            phase_type = np.array([{"P": 0, "S": 1}[x.upper()] for x in phase_type])
            event_index = np.array(event_index)
            station_index = np.concatenate(station_index)

            station_index = torch.tensor(station_index, dtype=torch.long)
            event_index = torch.tensor(event_index, dtype=torch.long)
            phase_weight = torch.tensor(phase_score, dtype=torch.float32)
            phase_time = torch.tensor(phase_time, dtype=torch.float32)
            phase_type = torch.tensor(phase_type, dtype=torch.long)

            meta[i] = {
                "event_index": event_index,
                "station_index": station_index,
                "phase_time": phase_time,
                "phase_weight": phase_weight,
                "phase_type": phase_type,
            }
        self.meta = meta

    def read_data_dd(self):
        meta = {}
        for i, index_batch in enumerate(self.event_index_batch):
            event_index = []
            station_index = []
            phase_score = []
            phase_time = []
            phase_type = []

            event_loc = self.events.loc[index_batch, ["x_km", "y_km", "z_km"]].values
            neighs = self.neigh.radius_neighbors(event_loc, sort_results=True)[1]

            for j, index1 in enumerate(index_batch):
                picks1 = self.picks_by_event.get_group(index1)

                for index2 in neighs[j][1 : self.max_pair_num + 1]:
                    if index1 >= index2:
                        continue
                    picks2 = self.picks_by_event.get_group(index2)
                    common = picks1.merge(picks2, on=["station_id", "phase_type"], how="inner")
                    phase_time.append(common["phase_time_x"].values - common["phase_time_y"].values)
                    phase_score.append((common["phase_score_x"].values + common["phase_score_y"].values) / 2.0)
                    phase_type.extend(common["phase_type"].values.tolist())
                    event_index.extend([[index1, index2]] * len(common))
                    station_index.append(self.stations.loc[common["station_id"], "index"].values)

            if len(phase_time) == 0:
                self.station_index = torch.tensor([], dtype=torch.long)
                self.event_index = torch.tensor([], dtype=torch.long)
                self.phase_weight = torch.tensor([], dtype=torch.float32)
                self.phase_time = torch.tensor([], dtype=torch.float32)
                self.phase_type = torch.tensor([], dtype=torch.long)
            elif len(phase_time) == 1:
                phase_type = np.array([{"P": 0, "S": 1}[x.upper()] for x in phase_type])
                event_index = np.array(event_index)
                self.station_index = torch.tensor(station_index, dtype=torch.long)
                self.event_index = torch.tensor(event_index, dtype=torch.long)
                self.phase_weight = torch.tensor(phase_score, dtype=torch.float32)
                self.phase_time = torch.tensor(phase_time, dtype=torch.float32)
                self.phase_type = torch.tensor(phase_type, dtype=torch.long)
            else:
                station_index = np.concatenate(station_index)
                phase_time = np.concatenate(phase_time)
                phase_score = np.concatenate(phase_score)
                phase_type = np.array([{"P": 0, "S": 1}[x.upper()] for x in phase_type])
                event_index = np.array(event_index)

                self.station_index = torch.tensor(station_index, dtype=torch.long)
                self.phase_time = torch.tensor(phase_time, dtype=torch.float32)
                self.phase_weight = torch.tensor(phase_score, dtype=torch.float32)
                self.phase_type = torch.tensor(phase_type, dtype=torch.long)
                self.event_index = torch.tensor(event_index, dtype=torch.long)

            meta[i] = {
                "event_index": self.event_index,
                "station_index": self.station_index,
                "phase_time": self.phase_time,
                "phase_weight": self.phase_weight,
                "phase_type": self.phase_type,
            }

        self.meta = meta

    def __getitem__(self, i):
        return self.meta[i]


class PhaseDatasetDD(Dataset):
    def __init__(
        self,
        pairs,
        picks,
        events,
        stations,
        batch_size=1000,
        config=None,
    ):
        self.pairs = pairs
        self.picks = picks
        self.events = events
        self.stations = stations
        self.config = config
        self.batch_size = batch_size

        self.idx_batch = np.array_split(np.arange(len(self.pairs)), (len(self.pairs) - 1) // self.batch_size + 1)

        print(f"Generated {len(self.pairs)} pairs")
        print(f"Split into {len(self.idx_batch)} batches of size {self.batch_size}")

    def __len__(self):
        return len(self.idx_batch)

    def __getitem__(self, i):

        idx = self.idx_batch[i]
        idx1_eve = self.pairs["event_index1"][idx]
        idx2_eve = self.pairs["event_index2"][idx]
        idx_eve = np.stack([idx1_eve, idx2_eve], axis=1)
        idx_sta = self.pairs["station_index"][idx]
        phase_weight = self.pairs["phase_score"][idx]
        phase_type = self.pairs["phase_type"][idx]
        phase_time = self.pairs["dd_time"][idx]

        return {
            "idx_eve": torch.tensor(idx_eve, dtype=torch.long),
            "idx_sta": torch.tensor(idx_sta, dtype=torch.long),
            "phase_type": torch.tensor(phase_type, dtype=torch.long),
            "phase_weight": torch.tensor(phase_weight, dtype=torch.float32),
            "phase_time": torch.tensor(phase_time, dtype=torch.float32),
        }
