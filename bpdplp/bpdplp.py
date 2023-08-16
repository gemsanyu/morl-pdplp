import os
import pathlib

import numpy as np

from bpdplp.utils import generate_graph, read_instance, read_road_types, generate_time_windows
from bpdplp.utils import RANDOM, RANDOMCLUSTER, CLUSTER, CENTRAL

TIME_HORIZONS = np.asanyarray([0,0.2,0.3,0.7,0.8,1000], dtype=np.float32)
SPEED_PROFILES = np.asanyarray([[1.5, 1, 1.67, 1.17, 1.33],[1.17, 0.67, 1.33, 0.83, 1],[1, 0.33, 0.67, 0.5, 0.83]], dtype=np.float32)

class BPDPLP(object):
    def __init__(self, 
                num_requests=10,
                num_vehicles=3,
                planning_time=120,
                time_window_length=60,
                max_capacity=100,
                graph_seed=None, 
                distribution=CLUSTER,
                depot_location=CENTRAL,
                cluster_delta=1.2,
                num_cluster=8,
                instance_name=None) -> None:
        self.num_requests = num_requests
        self.num_nodes = num_requests*2 + 1
        self.num_vehicles = num_vehicles
        self.planning_time = planning_time
        self.time_window_length = time_window_length
        self.max_capacity = max_capacity
        self.distribution = distribution
        self.depot_location = depot_location
        self.cluster_delta=cluster_delta
        self.num_cluster = num_cluster
        self.graph_seed = graph_seed
        self.instance_name = instance_name
        if instance_name is None:
            self.generate_instance()
        else:
            self.read_instance()
                 
        self.normalize()
            
    def read_instance(self):
        instance_path = pathlib.Path(".")/"dataset"/"test"/(self.instance_name+".txt")
        road_types_path = pathlib.Path(".")/"dataset"/"test"/(self.instance_name+".road_types")
        if os.path.isfile(instance_path.absolute()):
            instance = read_instance(instance_path)
            self.num_nodes, self.planning_time, self.max_capacity, self.coords, self.demands, self.time_windows, self.service_durations, self.distance_matrix = instance
            self.road_types = read_road_types(road_types_path, self.num_nodes)
        else:
            # if the instance is npz
            instance_path = pathlib.Path(".")/"dataset"/"test"/(self.instance_name+".npz")
            data = np.load(instance_path.absolute())
            self.num_nodes = data["num_nodes"]
            self.coords = data["coords"]
            self.norm_coords = data["norm_coords"]
            self.demands = data["demands"]
            self.norm_demands = data["norm_demands"]
            self.time_windows = data["time_windows"]
            self.norm_time_windows = data["norm_time_windows"]
            self.service_durations = data["service_durations"]
            self.norm_service_durations = data["norm_service_durations"]
            self.distance_matrix = data["distance_matrix"]
            self.norm_distance_matrix = data["norm_distance_matrix"]
            self.road_types = data["road_types"]
            self.planning_time = data["planning_time"]
            self.max_capacity = data["max_capacity"]

        #normalize all
    def normalize(self):
        self.norm_demands = self.demands / self.max_capacity
        min_coords, max_coords = np.min(self.coords, axis=0, keepdims=True), np.max(self.coords, axis=0, keepdims=True)
        self.norm_coords = (self.coords-min_coords)/(max_coords-min_coords)
        self.norm_time_windows = self.time_windows/self.planning_time
        self.norm_service_durations = self.service_durations/self.planning_time
        min_distance, max_distance = np.min(self.distance_matrix), np.max(self.distance_matrix)
        self.norm_distance_matrix = (self.distance_matrix-min_distance)/(max_distance-min_distance)
        
            
    """
    The L stands for list of candidate nodes as in (Sartori and Buriol, 2020)
    """
    def generate_instance(self):
        self.coords, self.distance_matrix = generate_graph(self.graph_seed, self.num_nodes, self.num_cluster, self.cluster_delta, self.distribution, self.depot_location)
        self.demands = np.random.random(size=(self.num_nodes))*(0.6*self.max_capacity-10) + 10
        self.demands = np.floor(self.demands)
        self.demands[0] = 0
        self.demands[self.num_requests+1:] = -self.demands[1:self.num_requests+1]
        self.service_durations = (np.random.randint(3, size=(self.num_nodes))+1)*5
        self.service_durations[0]=0
        self.time_windows = generate_time_windows(self.num_requests, self.planning_time, self.time_window_length, self.service_durations, self.distance_matrix)
        a = np.random.randint(0,3,size=(self.num_nodes, self.num_nodes), dtype=np.int8)
        road_types = np.tril(a) + np.tril(a, -1).T
        self.road_types = road_types
