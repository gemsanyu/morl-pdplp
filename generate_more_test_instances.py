from multiprocessing import Pool
import pathlib
import random

import numpy as np


from bpdplp.bpdplp import BPDPLP
from bpdplp.utils import RANDOM, RANDOMCLUSTER, CLUSTER, CENTRAL, read_graph

"""
This is different from generate instances,
this one randomly samples from the config
This is to generate additional test instances, other than
provided by sartori and buriol
useful for additional statistical evaluations,
so that from one evaluation to another,
the independence is not violated.
"""

def generate(nr,nv,nc,cd,pt,twl,mc,d,dl,graph_name,mode,idx):
    # nr,nv,nc,cd,pt,twl,mc,d,dl,graph_seed,mode,idx = config
    graph_seed = read_graph(graph_name)
    instance = BPDPLP(num_requests=nr,
                      num_vehicles=nv,
                      num_cluster=nc,
                      cluster_delta=cd,
                      planning_time=pt,time_window_length=twl,
                      max_capacity=mc,
                      distribution=d,
                      depot_location=dl,
                      graph_seed=graph_seed)
    graph_acronym = "new2-"+graph_name[:3]
    instance_name = graph_acronym+"-n"+str(nr)+"-"+str(idx)
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
                        max_capacity=instance.max_capacity,
                        planning_time=instance.planning_time,
                        num_nodes=instance.num_nodes)
    

def run():
    num_samples_per_config = 6
    num_requests_list = [100,200,400]
    num_vehicles_list = [1,3,5,10]
    num_clusters_list = [3,4,5,6,7,8]
    cluster_delta_list = [1,1.2,1.6]
    planning_time_list = [240,480]
    time_window_length_list = [60,120]
    max_capacity_list = [100,300]
    distribution_list = [RANDOM, RANDOMCLUSTER, CLUSTER]
    depot_location_list = [RANDOM, CENTRAL]
    graph_list = ["barcelona.txt", "berlin.txt", "poa.txt"]
    config_list = []
    for idx in range(num_samples_per_config):
        for nr in num_requests_list:
            for nv in num_vehicles_list:
                for graph in graph_list:
                    nc = num_clusters_list[random.randint(0, len(num_clusters_list)-1)]
                    pt = planning_time_list[random.randint(0, len(planning_time_list)-1)]
                    twl = time_window_length_list[random.randint(0, len(time_window_length_list)-1)]
                    cd = cluster_delta_list[random.randint(0, len(cluster_delta_list)-1)]
                    mc = max_capacity_list[random.randint(0, len(max_capacity_list)-1)]
                    d = distribution_list[random.randint(0, len(distribution_list)-1)]
                    dl = depot_location_list[random.randint(0, len(depot_location_list)-1)]
                    config_list += [(nr,nv,nc,cd,pt,twl,mc,d,dl,graph,"test",idx)]
    with Pool(processes=12) as pool:
        L = pool.starmap(generate, config_list)

if __name__ == "__main__":
    run()