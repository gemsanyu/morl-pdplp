from multiprocessing import Pool
import pathlib

import numpy as np

from bpdplp.bpdplp import BPDPLP
from bpdplp.utils import RANDOM, RANDOMCLUSTER, CLUSTER, CENTRAL, read_graph

def generate(nr,nv,nc,cd,pt,twl,mc,d,dl,graph_seed,mode,idx):
    # nr,nv,nc,cd,pt,twl,mc,d,dl,graph_seed,mode,idx = config
    instance = BPDPLP(num_requests=nr,
                      num_vehicles=nv,
                      num_cluster=nc,
                      cluster_delta=cd,
                      planning_time=pt,time_window_length=twl,
                      max_capacity=mc,
                      distribution=d,
                      depot_location=dl,
                      graph_seed=graph_seed)
    instance_name = "nr_"+str(nr)+"_nv_"+str(nv)+"_nc_"+str(nc)+"_cd_"+str(cd)+"_pt_"+str(pt)+"_twl_"+str(twl)+"_mc_"+str(mc)+"_d_"+str(d)+"_dl_"+str(dl)+"_idx_"+str(idx)
    data_dir = pathlib.Path(".")/"dataset"/mode
    data_dir.mkdir(parents=True,exist_ok=True)
    data_path = data_dir/(instance_name+".npz")
    
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
    num_samples_per_config = 1
    num_requests_list = [50]
    num_vehicles_list = [1,2,3,5]
    num_clusters_list = [3,4,5,6,7,8]
    cluster_delta_list = [1,1.2,1.6]
    planning_time_list = [240,480]
    time_window_length_list = [60,120]
    max_capacity_list = [100,300]
    distribution_list = [RANDOM]
    depot_location_list = [RANDOM, CENTRAL]
    mode ="training"
    graph_seed = read_graph("barcelona.txt")
    config_list = [(nr,nv,nc,cd,pt,twl,mc,d,dl,graph_seed,mode,idx) for nr in num_requests_list for nv in num_vehicles_list for nc in num_clusters_list for cd in cluster_delta_list for pt in planning_time_list for twl in time_window_length_list for mc in max_capacity_list for d in distribution_list for dl in depot_location_list for idx in range(num_samples_per_config)]
    with Pool(processes=12) as pool:
        L = pool.starmap(generate, config_list)

if __name__ == "__main__":
    run()