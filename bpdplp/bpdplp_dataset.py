import pathlib
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from bpdplp.utils import RANDOM, RANDOMCLUSTER, CLUSTER, CENTRAL, read_graph

class BPDPLP_Dataset(Dataset):
    def __init__(self,
                 num_samples:int=1000000,
                 num_requests:int = 50,
                 num_vehicles_list:List[int] = [1,3,5,10],
                 num_clusters_list:List[int] = [3,4,5,6,7,8],
                 cluster_delta_list:List[float] = [1,1.2,1.6],
                 planning_time_list:List[int] = [240,480],
                 time_window_length_list:List[int] = [60,120],
                 max_capacity_list:List[int] = [100,300],
                 distribution_list:List[int] = [RANDOM, RANDOMCLUSTER, CLUSTER],
                 depot_location_list:List[int] = [RANDOM, CENTRAL],
                 graph_seed_name_list:List[str] = ["barcelona.txt"],
                 mode:str="training",
            ) -> None:
        super(BPDPLP_Dataset, self).__init__()

        # num_vehicles_list = [1,2,3,5]
        # num_clusters_list = [3]
        # cluster_delta_list = [1]
        # planning_time_list = [240]
        # time_window_length_list = [60]
        # max_capacity_list = [100]
        # distribution_list = [RANDOM]
        # depot_location_list = [RANDOM]

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
        # self.graph_seed_list = [read_graph(graph_seed_name) for graph_seed_name in graph_seed_name_list]
        self.config_list = [(num_requests,nv,nc,cd,pt,twl,mc,d,dl) for nv in num_vehicles_list for nc in num_clusters_list for cd in cluster_delta_list for pt in planning_time_list for twl in time_window_length_list for mc in max_capacity_list for d in distribution_list for dl in depot_location_list]
        self.mode = mode
        
        # self.coords_list = []
        # self.norm_coords_list = []
        # self.demands_list = []
        # self.norm_demands_list = []
        # self.time_windows_list = []
        # self.norm_time_windows_list = []
        # self.service_durations_list = []
        # self.norm_service_durations_list = []
        # self.distance_matrix_list = []
        # self.norm_distance_matrix_list = []
        # self.road_types_list = []

        # for index in range(self.num_samples):
        #     config = self.config_list[index%len(self.config_list)]
        #     nr,nv,nc,cd,pt,twl,mc,d,dl = config
        #     idx = (index // len(self.config_list))
        #     instance_name = "nr_"+str(nr)+"_nv_"+str(nv)+"_nc_"+str(nc)+"_cd_"+str(cd)+"_pt_"+str(pt)+"_twl_"+str(twl)+"_mc_"+str(mc)+"_d_"+str(d)+"_dl_"+str(dl)+"_idx_"+str(idx)
        #     data_path = pathlib.Path(".")/"dataset"/self.mode/(instance_name+".npz")
        #     data = np.load(data_path.absolute())
        #     coords = torch.from_numpy(data["coords"])
        #     norm_coords = torch.from_numpy(data["norm_coords"])
        #     demands = torch.from_numpy(data["demands"])
        #     norm_demands = torch.from_numpy(data["norm_demands"])
        #     time_windows = torch.from_numpy(data["time_windows"])
        #     norm_time_windows = torch.from_numpy(data["norm_time_windows"])
        #     service_durations = torch.from_numpy(data["service_durations"])
        #     norm_service_durations = torch.from_numpy(data["norm_service_durations"])
        #     distance_matrix = torch.from_numpy(data["distance_matrix"])
        #     norm_distance_matrix = torch.from_numpy(data["norm_distance_matrix"])
        #     road_types = torch.from_numpy(data["road_types"])
        #     self.coords_list += [coords]
        #     self.norm_coords_list += [norm_coords]
        #     self.demands_list += [demands]
        #     self.norm_demands_list += [norm_demands]
        #     self.time_windows_list += [time_windows]
        #     self.norm_time_windows_list += [norm_time_windows]
        #     self.service_durations_list += [service_durations]
        #     self.norm_service_durations_list += [norm_service_durations]
        #     self.distance_matrix_list += [distance_matrix]
        #     self.norm_distance_matrix_list += [norm_distance_matrix]
        #     self.road_types_list += [road_types]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        config = self.config_list[index%len(self.config_list)]
        nr,nv,nc,cd,pt,twl,mc,d,dl = config
        idx = (index // len(self.config_list))
        instance_name = "nr_"+str(nr)+"_nv_"+str(nv)+"_nc_"+str(nc)+"_cd_"+str(cd)+"_pt_"+str(pt)+"_twl_"+str(twl)+"_mc_"+str(mc)+"_d_"+str(d)+"_dl_"+str(dl)+"_idx_"+str(idx)
        data_path = pathlib.Path(".")/"dataset"/self.mode/(instance_name+".npz")
        data = np.load(data_path.absolute())
        # coords = self.coords_list[index]
        # norm_coords =  self.norm_coords_list[index]
        # demands =  self.demands_list[index]
        # norm_demands =  self.norm_demands_list[index]
        # time_windows =  self.time_windows_list[index]
        # norm_time_windows =  self.norm_time_windows_list[index]
        # service_durations =  self.service_durations_list[index]
        # norm_service_durations = self.norm_service_durations_list[index]
        # distance_matrix = self.distance_matrix_list[index]
        # norm_distance_matrix = self.norm_distance_matrix_list[index]
        # road_types = self.road_types_list[index]
        coords = torch.from_numpy(data["coords"])
        norm_coords = torch.from_numpy(data["norm_coords"])
        demands = torch.from_numpy(data["demands"])
        norm_demands = torch.from_numpy(data["norm_demands"])
        time_windows = torch.from_numpy(data["time_windows"])
        norm_time_windows = torch.from_numpy(data["norm_time_windows"])
        service_durations = torch.from_numpy(data["service_durations"])
        norm_service_durations = torch.from_numpy(data["norm_service_durations"])
        distance_matrix = torch.from_numpy(data["distance_matrix"])
        norm_distance_matrix = torch.from_numpy(data["norm_distance_matrix"])
        road_types = torch.from_numpy(data["road_types"])
        max_capacity = mc
        return index, nv, max_capacity, coords, norm_coords, demands, norm_demands, pt, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types
