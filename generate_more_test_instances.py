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

def generate(nr,nc,cd,pt,twl,mc,d,dl,graph_name,mode,idx):
    # nr,nv,nc,cd,pt,twl,mc,d,dl,graph_seed,mode,idx = config
    graph_seed = read_graph(graph_name)
    instance = BPDPLP(num_requests=nr,
                      num_vehicles=1,
                      num_cluster=nc,
                      cluster_delta=cd,
                      planning_time=pt,time_window_length=twl,
                      max_capacity=mc,
                      distribution=d,
                      depot_location=dl,
                      graph_seed=graph_seed)
    graph_acronym = "new2-"+graph_name[:3]
    instance_name = graph_acronym+"-n"+str(nr*2)+"-"+str(idx)
    data_dir = pathlib.Path(".")/"dataset"/mode
    data_dir.mkdir(parents=True,exist_ok=True)
    save_to_text(instance_name, instance, graph_name)
    
def save_to_text(instance_name, instance:BPDPLP, graph_name):
    data_dir = pathlib.Path(".")/"dataset"/"test"
    data_dir.mkdir(parents=True,exist_ok=True)
    instance_path = data_dir/(instance_name+".txt")
    road_types_path = data_dir/(instance_name+".road_types")
    with open(instance_path.absolute(), "w") as instance_file:
        instance_file.write("NAME: "+instance_name+"\n")
        instance_file.write("LOCATION: "+graph_name+"\n")
        instance_file.write("COMMENT: GENERATED based on Sartori and Buriol (2019)\n")
        instance_file.write("TYPE: PDPTW\n")
        instance_file.write("SIZE: "+str(instance.num_nodes)+"\n")
        instance_file.write("DISTRIBUTION: placeholder\n")
        instance_file.write("DEPOT: placeholder\n")
        instance_file.write("ROUTE-TIME: "+str(instance.planning_time)+"\n")
        instance_file.write("TIME-WINDOW: "+str(instance.time_window_length)+"\n")
        instance_file.write("CAPACITY: "+str(instance.max_capacity)+"\n")
        instance_file.write("NODES\n")
        for i in range(instance.num_nodes):
            line = str(i) + " "
            line += f'{instance.coords[i,0]:.8f} {instance.coords[i,1]:.8f}' + " "
            line += str(int(instance.demands[i])) + " "
            line += str(int(instance.time_windows[i,0])) + " " + str(int(instance.time_windows[i,1])) + " "
            line += str(int(instance.demands[i])) + " "
            if i == 0:
                line += "0 0\n"
            elif i <= instance.num_requests:
                line += "0 " + str(i+instance.num_requests) + "\n"
            else:
                line += str(i-instance.num_requests) + " 0\n"
            instance_file.write(line)
        instance_file.write("EDGES\n")       
        for i in range(instance.num_nodes):
            line = ""
            for j in range(instance.num_nodes):
                line += str(int(instance.distance_matrix[i,j])) + " "
            line += "\n"
            instance_file.write(line)
        instance_file.write("EOF")

    """
        saving road types
    """
    with open(road_types_path.absolute(), "w") as road_types_file:
        for i in range(instance.num_nodes):
            line = ""
            for j in range(instance.num_nodes):
                line += str(instance.road_types[i,j])+" "
            road_types_file.write(line+"\n")


def run():
    num_samples_per_config = 6
    num_requests_list = [50,100,200]
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
            for graph in graph_list:
                nc = num_clusters_list[random.randint(0, len(num_clusters_list)-1)]
                pt = planning_time_list[random.randint(0, len(planning_time_list)-1)]
                twl = time_window_length_list[random.randint(0, len(time_window_length_list)-1)]
                cd = cluster_delta_list[random.randint(0, len(cluster_delta_list)-1)]
                mc = max_capacity_list[random.randint(0, len(max_capacity_list)-1)]
                d = distribution_list[random.randint(0, len(distribution_list)-1)]
                dl = depot_location_list[random.randint(0, len(depot_location_list)-1)]
                config_list += [(nr,nc,cd,pt,twl,mc,d,dl,graph,"test",idx+1)]
    with Pool(processes=12) as pool:
        L = pool.starmap(generate, config_list)

if __name__ == "__main__":
    run()