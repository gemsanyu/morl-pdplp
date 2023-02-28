import pathlib
import pickle
import random
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

from bpdplp.bpdplp import BPDPLP
from bpdplp.utils import RANDOM, RANDOMCLUSTER, CLUSTER, CENTRAL, read_graph

class BPDPLP_Dataset(Dataset):
    def __init__(self,
                 num_samples:int=1000000,
                 num_requests:int = 50,
                 num_vehicles_list:List[int] = [1,2,3,5],
                 num_clusters_list:List[int] = [3,4,5,6,7,8],
                 cluster_delta_list:List[float] = [1,1.2,1.6],
                 planning_time_list:List[int] = [240,480],
                 time_window_length_list:List[int] = [60,120],
                 max_capacity_list:List[int] = [100,300],
                 distribution_list:List[int] = [RANDOM, RANDOMCLUSTER, CLUSTER],
                 depot_location_list:List[int] = [RANDOM, CENTRAL],
                 graph_seed_name_list:List[str] = ["barcelona.txt"]
            ) -> None:
        super(BPDPLP_Dataset, self).__init__()

        self.num_samples = num_samples
        self.num_requests = num_requests
        self.num_vehicles_list = num_vehicles_list
        self.num_clusters_list = num_clusters_list
        self.cluster_delta_list = cluster_delta_list
        self.planning_time_list = planning_time_list
        self.time_window_length_list = time_window_length_list
        self.max_capacity_list = max_capacity_list
        self.distribution_list = distribution_list
        self.depot_location_list = depot_location_list
        self.graph_seed_list = [read_graph(graph_seed_name) for graph_seed_name in graph_seed_name_list]
        self.config_list = [(num_requests,nv,nc,cd,pt,twl,mc,d,dl,graph_seed) for nv in num_vehicles_list for nc in num_clusters_list for cd in cluster_delta_list for pt in planning_time_list for twl in time_window_length_list for mc in max_capacity_list for d in distribution_list for dl in depot_location_list for graph_seed in self.graph_seed_list]


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        config = self.config_list[index%len(self.config_list)]
        nr,nv,nc,cd,pt,twl,mc,d,dl,graph_seed = config
        instance = BPDPLP(num_requests=nr,
                          num_vehicles=nv,
                          num_cluster=nc,
                          cluster_delta=cd,
                          planning_time=pt,
                          time_window_length=twl,
                          max_capacity=mc,
                          distribution=d,
                          depot_location=dl,
                          graph_seed=graph_seed)
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
        max_capacity = instance.max_capacity
        return nv, max_capacity, coords, norm_coords, demands, norm_demands, pt, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types
