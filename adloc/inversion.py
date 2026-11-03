import torch
import torch.optim as optim
from tqdm import tqdm


def optimize_dd(data_loader, travel_time, config):

    # init loss
    loss = 0
    for meta in data_loader:
        station_index = meta["idx_sta"]
        event_index = meta["idx_eve"]
        phase_time = meta["phase_time"]
        phase_type = meta["phase_type"]
        phase_weight = meta["phase_weight"]

        loss += travel_time(
            station_index,
            event_index,
            phase_type,
            phase_time,
            phase_weight,
        )["loss"]

    # if args.distributed:
    #     dist.barrier()
    #     dist.all_reduce(loss)
    print(f"Init loss: {loss}")

    ## invert loss
    # optimizer = optim.Adam(params=travel_time.parameters(), lr=0.1)
    # EPOCHS = 3
    # for i in range(EPOCHS):
    #     loss = 0
    #     optimizer.zero_grad()
    #     for meta in tqdm(data_loader, desc=f"Epoch {i}"):
    #         station_index = meta["idx_sta"]
    #         event_index = meta["idx_eve"]
    #         phase_time = meta["phase_time"]
    #         phase_type = meta["phase_type"]
    #         phase_weight = meta["phase_weight"]

    #         loss_ = travel_time(
    #             station_index,
    #             event_index,
    #             phase_type,
    #             phase_time,
    #             phase_weight,
    #         )["loss"]

    #         # optimizer.zero_grad()
    #         loss_.backward()
    #         # optimizer.step()
    #         loss += loss_

    #     optimizer.step()
    #     print(f"Loss: {loss}")

    optimizer = optim.LBFGS(params=travel_time.parameters(), max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = 0
        for meta in tqdm(data_loader, desc=f"BFGS"):
            station_index = meta["idx_sta"]
            event_index = meta["idx_eve"]
            phase_time = meta["phase_time"]
            phase_type = meta["phase_type"]
            phase_weight = meta["phase_weight"]

            loss_ = travel_time(
                station_index,
                event_index,
                phase_type,
                phase_time,
                phase_weight,
            )["loss"]
            loss_.backward()
            loss += loss_

        print(f"Loss: {loss}")
        # if args.distributed:
        #     dist.barrier()
        #     dist.all_reduce(loss)

        return loss

    optimizer.step(closure)

    # travel_time.event_loc.weight.grad.data -= travel_time.event_loc.weight.grad.data.mean()
    # travel_time.event_time.weight.grad.data -= travel_time.event_time.weight.grad.data.mean()

    ## updated loss
    loss = 0
    for meta in data_loader:
        station_index = meta["idx_sta"]
        event_index = meta["idx_eve"]
        phase_time = meta["phase_time"]
        phase_type = meta["phase_type"]
        phase_weight = meta["phase_weight"]

        loss += travel_time(
            station_index,
            event_index,
            phase_type,
            phase_time,
            phase_weight,
        )["loss"]

    # if args.distributed:
    #     dist.barrier()
    #     dist.all_reduce(loss)

    print(f"Invert loss: {loss}")

    # set variable range
    # travel_time.event_loc.weight.data[:, 2] += (
    #     (torch.rand_like(travel_time.event_loc.weight.data[:, 2]) - 0.5) * (args.epochs - 1 - i) / args.epochs
    # )
    travel_time.event_loc.weight.data[:, 2].clamp_(min=config["zlim_km"][0], max=config["zlim_km"][1])
