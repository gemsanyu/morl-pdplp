import pathlib
import pickle
import random
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

from bpdplp.bpdplp import BPDPLP
from bpdplp.utils import RANDOM, RANDOMCLUSTER, CLUSTER, CENTRAL

class BPDPLP_Dataset(Dataset):
    def __init__(self,
                 num_samples:int=1000000,
                 num_requests:int = 50,
                 num_vehicles:int = 2,
                 num_clusters_list:List[int] = [3,4,5,6,7,8],
                 cluster_delta_list:List[float] = [1,1.2,1.6],
                 planning_time_list:List[int] = [240,480],
                 time_window_length_list:List[int] = [60,120],
                 max_capacity_list:List[int] = [100,300],
                 distribution_list:List[int] = [RANDOM, RANDOMCLUSTER, CLUSTER],
                 depot_location_list:List[int] = [RANDOM, CENTRAL],
            ) -> None:
        super(BPDPLP_Dataset, self).__init__()

        self.num_samples = num_samples
        self.num_requests = num_requests
        self.num_vehicles = num_vehicles
        self.num_clusters_list = num_clusters_list
        self.cluster_delta_list = cluster_delta_list
        self.planning_time_list = planning_time_list
        self.time_window_length_list = time_window_length_list
        self.max_capacity_list = max_capacity_list
        self.distribution_list = distribution_list
        self.depot_location_list = depot_location_list

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        num_requests = self.num_requests
        num_vehicles = self.num_vehicles
        num_clusters = random.choice(self.num_clusters_list)
        cluster_delta = random.choice(self.cluster_delta_list)
        planning_time = random.choice(self.planning_time_list)
        time_window_length = random.choice(self.time_window_length_list)
        max_capacity = random.choice(self.max_capacity_list)
        distribution = random.choice(self.distribution_list)
        depot_location = random.choice(self.depot_location_list)
        instance = BPDPLP(num_requests, num_vehicles, planning_time, time_window_length, max_capacity, distribution=distribution, depot_location=depot_location, cluster_delta=cluster_delta, num_cluster=num_clusters)
        
        coords = torch.from_numpy(instance.coords)
        norm_coords = torch.from_numpy(instance.norm_coords)
        demands = torch.from_numpy(instance.demands)
        norm_demands = torch.from_numpy(instance.norm_demands)
        time_windows = torch.from_numpy(instance.time_windows)
        norm_time_windows = torch.from_numpy(instance.norm_time_windows)
        service_durations = torch.from_numpy(instance.service_durations)
        norm_service_durations = torch.from_numpy(instance.norm_service_durations)
        distance_matrix = torch.from_numpy(instance.distance_matrix)
        norm_distance_matrix = torch.from_numpy(instance.norm_distance_matrix)
        road_types = torch.from_numpy(instance.road_types)
        return coords, norm_coords, demands, norm_demands, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types
