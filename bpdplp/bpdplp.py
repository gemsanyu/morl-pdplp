import os
import pathlib

import numpy as np

class BPDPLP(object):
    def __init__(self, 
                num_requests=10,
                num_vehicles=3,
                planning_time=240,
                graph_seed="barcelona.txt", 
                instance_name=None) -> None:
        self.num_requests = num_requests
        self.num_nodes = num_requests*2 + 1
        self.num_vehicles = num_vehicles
        self.planning_time = planning_time

        self.graph_seed = graph_seed
        self.instance_name = instance_name
        if instance_name is None:
            self.generate_instance()
        else:
            self.read_instance()
            
    def read_instance(self):
        instance_path = pathlib.Path(".")/"dataset"/(self.instance_name+".txt")
        road_type_path = pathlib.Path(".")/"dataset"/(self.instance_name+".road_types")
        with open(instance_path.absolute(), "r") as instance_file:
            lines = instance_file.readlines()
            for i in range(11):
                strings = lines[i].split()
                if i == 4:
                    self.num_nodes = int(strings[1])
                elif i == 7:
                    self.planning_time = int(strings[1])
                elif i == 9:
                    capacity = float((strings[1]))
                    self.max_capacity = np.asanyarray([capacity]*self.num_vehicles)
            self.coords = np.zeros((self.num_nodes,2), dtype=np.float32)
            self.demands = np.zeros((self.num_nodes,), dtype=np.float32)
            self.time_windows = np.zeros((self.num_nodes,2), dtype=np.float32)
            self.service_durations = np.zeros((self.num_nodes,), dtype=np.float32)
            for i in range(11, self.num_nodes+11):
                strings = lines[i].split()
                idx = i-11
                self.coords[idx,0], self.coords[idx,1] = float(strings[1]), float(strings[2])
                self.demands[idx] = float(strings[3])
                self.time_windows[idx,0], self.time_windows[idx,1] = strings[4], strings[5]
                self.service_durations[idx] = strings[6]
            self.distance_matrix = np.zeros((self.num_nodes,self.num_nodes), dtype=np.float32)
            for i in range(self.num_nodes+12, 2*self.num_nodes+12):
                strings = lines[i].split()
                idx = i-(self.num_nodes+12)
                for j in range(self.num_nodes):
                    self.distance_matrix[idx,j] = float(strings[j])
        road_types = np.zeros((self.num_nodes,self.num_nodes), dtype=np.int8)
        if not os.path.isfile(road_type_path.absolute()):
            a = np.random.randint(0,3,size=(self.num_nodes,self.num_nodes), dtype=np.int8)
            road_types = np.tril(a) + np.tril(a, -1).T
            with open(road_type_path.absolute(), "w") as road_types_file:
                for i in range(self.num_nodes):
                    line = ""
                    for j in range(self.num_nodes):
                       line += str(road_types[i,j])+" "
                    road_types_file.write(line+"\n")
        else:
            with open(road_type_path.absolute(), "r") as road_types_file:
                lines = road_types_file.readlines()
                for i,line in enumerate(lines):
                    strings = line.split()
                    for j in range(self.num_nodes):
                       road_types[i,j] = int(strings[j])
            
        #normalize all
        self.norm_demands = self.demands / self.max_capacity[0]
        min_coords, max_coords = np.min(self.coords, axis=0, keepdims=True), np.max(self.coords, axis=0, keepdims=True)
        self.norm_coords = (self.coords-min_coords)/(max_coords-min_coords)
        self.norm_time_windows = self.time_windows/self.planning_time
        self.norm_service_durations = self.service_durations/self.planning_time
        min_distance, max_distance = np.min(self.distance_matrix), np.max(self.distance_matrix)
        self.norm_distance_matrix = (self.distance_matrix-min_distance)/(max_distance-min_distance)
        
                
                
                      
                

    def generate_instance(self):
        graph_path = pathlib.Path(".")/"dataset"/"graphs"/self.graph_seed
    #     with open(graph_path.absolute(), "r") as graph_file:
            