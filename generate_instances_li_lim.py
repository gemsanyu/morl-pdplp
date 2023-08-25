from multiprocessing import Pool
import pathlib

import numpy as np

from bpdplp.bpdplp import BPDPLP
from bpdplp.utils import RANDOM, RANDOMCLUSTER, CLUSTER, CENTRAL, read_graph

SMALL = 0
BIG = 1

def generate(nr,nv,nc,cd,it,twl,d,dl,mode,idx):
    # nr,nv,nc,cd,pt,twl,mc,d,dl,graph_seed,mode,idx = config
    if d == CLUSTER:
        if it == SMALL:
            pt = 1236
            mc = 200
        else:
            pt = 3390
            mc = 700
    elif d == RANDOM:
        if it == SMALL:
            pt = 200
            mc = 230
        else:
            pt = 1000
            mc = 1000
    else:
        if it == SMALL:
            pt = 200
            mc = 240
        else:
            pt = 1000
            mc = 960
        # if pt_type == SHORT:
        # else:
        
        # if mc_type == SMALL:
        # else:
        
    instance = BPDPLP(num_requests=nr,
                      num_vehicles=nv,
                      num_cluster=nc,
                      cluster_delta=cd,
                      planning_time=pt,
                      time_window_length=twl,
                      max_capacity=mc,
                      distribution=d,
                      depot_location=dl,
                      graph_seed=None,
                      li_lim=True)
    instance_name = "lim_instances/nr_"+str(nr)+"_nv_"+str(nv)+"_nc_"+str(nc)+"_cd_"+str(cd)+"_it_"+str(it)+"_twl_"+str(twl)+"_d_"+str(d)+"_dl_"+str(dl)+"_idx_"+str(idx)
    data_dir = pathlib.Path(".")/"dataset"/mode
    data_dir.mkdir(parents=True,exist_ok=True)
    data_path = data_dir/(instance_name+".npz")
    data_path.parents[0].mkdir(parents=True,exist_ok=True)
    
    np.savez_compressed(data_path.absolute(), 
                        coords=instance.coords,
                        norm_coords=instance.coords,
                        demands=instance.demands,
                        norm_demands=instance.norm_demands,
                        time_windows=instance.time_windows,
                        norm_time_windows=instance.norm_time_windows,
                        service_durations=instance.service_durations,
                        norm_service_durations=instance.norm_service_durations,
                        distance_matrix=instance.distance_matrix,
                        norm_distance_matrix=instance.norm_distance_matrix,
                        road_types=instance.road_types,
                        max_capacity=instance.max_capacity)
    

def run():
    num_samples_per_config = 40
    num_requests_list = [50]
    num_vehicles_list = [1,2,3,5]
    num_clusters_list = [3,4,5,6,7,8]
    cluster_delta_list = [1,1.2,1.6]
    time_window_length_list = [60, 120]
    instance_type_list = [SMALL, BIG]
    distribution_list = [RANDOM, RANDOMCLUSTER, CLUSTER]
    depot_location_list = [RANDOM, CENTRAL]
    mode ="training"
    config_list = [(nr,nv,nc,cd,it,twl,d,dl,mode,idx) for nr in num_requests_list for nv in num_vehicles_list for nc in num_clusters_list for cd in cluster_delta_list for it in instance_type_list for twl in time_window_length_list for d in distribution_list for dl in depot_location_list for idx in range(num_samples_per_config)]
    with Pool(processes=12) as pool:
        L = pool.starmap(generate, config_list)

if __name__ == "__main__":
    run()