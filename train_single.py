import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from bpdplp.bpdplp_env import BPDPLP_Env
from bpdplp.bpdplp_dataset import BPDPLP_Dataset

def run():
    dataset = BPDPLP_Dataset(num_requests=10, num_vehicles_list=[1,2])
    dl = DataLoader(dataset, batch_size=2)
    for i, batch in enumerate(dl):
        num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types = batch
        env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
        static_features, vehicle_dynamic_features, customer_static_features, feasibility_mask = env.begin()
        # print(static_features)
        exit()

if __name__ == "__main__":
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    run()