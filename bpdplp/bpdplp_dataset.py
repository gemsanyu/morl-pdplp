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
                 num_vehicles_list:List[int] = [1,2,3,5],
                 num_clusters_list:List[int] = [3,4,5,6,7,8],
                 cluster_delta_list:List[float] = [1,1.2,1.6],
                 planning_time_list:List[int] = [240,480],
                 time_window_length_list:List[int] = [60,120],
                 max_capacity_list:List[int] = [100,300],
                 distribution_list:List[int] = [RANDOM, RANDOMCLUSTER, CLUSTER],
                 depot_location_list:List[int] = [RANDOM, CENTRAL],
                 graph_seed_name_list:List[str] = ["barcelona.txt"],
                 mode:str="training",
                 li_lim=False,
            ) -> None:
        super(BPDPLP_Dataset, self).__init__()
        
        self.li_lim = li_lim
        self.num_samples = num_samples
        self.num_requests = num_requests
            
        if li_lim:
            self.num_requests = 50
            self.num_vehicles_list = [1,2,3,5]
            self.num_clusters_list = [3,4,5,6,7,8]
            self.cluster_delta_list = [1,1.2,1.6]
            self.time_window_length_list = [60, 120]
            self.instance_type_list = [0, 1]
            self.distribution_list = [RANDOM, RANDOMCLUSTER, CLUSTER]
            self.depot_location_list = [RANDOM, CENTRAL]
            self.config_list = [(num_requests,nv,nc,cd,it,twl,d,dl) for it in self.instance_type_list for nv in num_vehicles_list for nc in num_clusters_list for cd in cluster_delta_list for pt in planning_time_list for twl in time_window_length_list for mc in max_capacity_list for d in distribution_list for dl in depot_location_list]
        else:
            self.num_vehicles_list = num_vehicles_list
            self.num_clusters_list = num_clusters_list
            self.cluster_delta_list = cluster_delta_list
            self.planning_time_list = planning_time_list
            self.time_window_length_list = time_window_length_list
            self.max_capacity_list = max_capacity_list
            self.distribution_list = distribution_list
            self.depot_location_list = depot_location_list
            self.config_list = [(num_requests,nv,nc,cd,pt,twl,mc,d,dl) for nv in num_vehicles_list for nc in num_clusters_list for cd in cluster_delta_list for pt in planning_time_list for twl in time_window_length_list for mc in max_capacity_list for d in distribution_list for dl in depot_location_list]
        
        
        self.mode = mode
        

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        config = self.config_list[index%len(self.config_list)]
        idx = (index // len(self.config_list))
        
        if self.li_lim:
            nr,nv,nc,cd,it,twl,d,dl = config
            instance_name = "lim_instances/nr_"+str(nr)+"_nv_"+str(nv)+"_nc_"+str(nc)+"_cd_"+str(cd)+"_it_"+str(it)+"_twl_"+str(twl)+"_d_"+str(d)+"_dl_"+str(dl)+"_idx_"+str(idx)
        else:
            nr,nv,nc,cd,pt,twl,mc,d,dl = config
            instance_name = "nr_"+str(nr)+"_nv_"+str(nv)+"_nc_"+str(nc)+"_cd_"+str(cd)+"_pt_"+str(pt)+"_twl_"+str(twl)+"_mc_"+str(mc)+"_d_"+str(d)+"_dl_"+str(dl)+"_idx_"+str(idx)
        data_path = pathlib.Path(".")/"dataset"/self.mode/(instance_name+".npz")
        data = np.load(data_path.absolute())
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
        max_capacity = int(data["max_capacity"])
        planning_time = int(data["planning_time"])
        return index, nv, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types
