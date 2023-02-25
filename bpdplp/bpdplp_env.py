import time

import numpy as np

from bpdplp.bpdplp import TIME_HORIZONS, SPEED_PROFILES

def compute_travel_time_vectorized(distances, current_times, time_horizons, speed_profiles):
    num_pair = distances.shape[0]
    pair_idx = np.arange(num_pair)
    horizons = np.argmax(time_horizons > current_times[:, np.newaxis], axis=1)-1
    temp_times = current_times
    is_distance_nonzero = distances > 0
    i = 0
    while np.any(distances>0):
        arrived_time = temp_times + distances/speed_profiles[pair_idx, horizons]
        is_arrived_time_pass_breakpoint = arrived_time > time_horizons[pair_idx, horizons+1]
        #is_nonzero_and_not_pass_breakpoint
        inanpb = np.logical_and(is_distance_nonzero, np.logical_not(is_arrived_time_pass_breakpoint))
        distances[inanpb] = 0
        #is_nonzero_and_pass_breakpoint
        inapb = np.logical_and(is_distance_nonzero, is_arrived_time_pass_breakpoint)
        distances[inapb] -= speed_profiles[pair_idx[inapb], horizons[inapb]]*(time_horizons[pair_idx[inapb],horizons[inapb]+1]-temp_times[inapb])
        temp_times[inapb] = time_horizons[pair_idx[inapb],horizons[inapb]+1]
        horizons[inapb] += 1
    travel_time = arrived_time - current_times
    return travel_time

def compute_travel_time(distance, current_time, time_horizon, speed_profile):
    horizon = np.searchsorted(time_horizon, current_time) - 1
    # horizon = 0
    # while time_horizons[horizon+1]<current_time:
    #     horizon += 1
    temp_time = current_time
    arrived_time = temp_time 
    while distance > 0:
        arrived_time = temp_time + distance/speed_profile[horizon]
        if arrived_time > time_horizon[horizon+1]:
            distance -= speed_profile[horizon]*(time_horizon[horizon+1]-temp_time)
            temp_time = time_horizon[horizon+1]
            horizon+=1
        else:
            distance=0
    travel_time = arrived_time-current_time
    return travel_time    


class BPDPLP_Env(object):
    def __init__(self, 
                 num_vehicles, 
                 max_capacity, 
                 coords, 
                 norm_coords, 
                 demands, 
                 norm_demands,
                 planning_time, 
                 time_windows, 
                 norm_time_windows, 
                 service_durations, 
                 norm_service_durations, 
                 distance_matrix, 
                 norm_distance_matrix, 
                 road_types) -> None:
        
        self.num_vehicles = num_vehicles.numpy()
        self.num_vehicles_cum = np.insert(np.cumsum(self.num_vehicles),0,0)
        self.max_capacity = max_capacity.numpy()
        self.batch_size, self.num_nodes, _ = coords.shape
        self.num_requests = (self.num_nodes-1)//2
        self.coords = coords.numpy()
        self.norm_coords = norm_coords.numpy()
        self.demands = demands.numpy()
        self.norm_demands = norm_demands.numpy()
        self.planning_time = planning_time.numpy()
        self.time_windows = time_windows.numpy()
        self.norm_time_windows = norm_time_windows.numpy()
        self.service_durations = service_durations.numpy()
        self.norm_service_durations = norm_service_durations.numpy()
        self.distance_matrix = distance_matrix.numpy()
        self.norm_distance_matrix = norm_distance_matrix.numpy()
        self.road_types = road_types.numpy()
        self.reset()

    def reset(self):
        self.current_load = [np.asanyarray([0]*self.num_vehicles[i], dtype=np.float32) for i in range(self.batch_size)]
        self.current_time = [np.asanyarray([0]*self.num_vehicles[i], dtype=np.float32) for i in range(self.batch_size)]
        self.current_location_idx = [np.asanyarray([0]*self.num_vehicles[i], dtype=int) for i in range(self.batch_size)]
        self.is_node_visited = np.zeros((self.batch_size, self.num_nodes), dtype=bool)
        self.is_node_visited[:, 0] = True
        self.request_assignment = np.full((self.batch_size, self.num_requests), -1, dtype=int)
        self.distance_travelled = [[0]*self.num_vehicles[i] for i in range(self.batch_size)]
        self.late_penalty = [0]*self.batch_size
        self.tour_list = [[[]]*self.num_vehicles[i] for i in range(self.batch_size)]
        self.departure_time_list = [[[]]*self.num_vehicles[i] for i in range(self.batch_size)]
               


    def begin(self):
        self.reset()
        return self.static_features, self.vehicle_dynamic_features, self.node_dynamic_features, self.feasibility_mask

    """
    coords, demands, service_durations, time_windows
    """
    @property
    def static_features(self):
        features = np.zeros((self.batch_size, self.num_nodes, 6), dtype=np.float32)
        features[:,:,:2] = self.norm_coords
        features[:,:,2] = self.norm_demands
        features[:,:,3] = self.norm_service_durations
        features[:,:,4:] = self.norm_time_windows
        return features
    
    """
        current_coords, current_load, current_time
    """
    @property
    def vehicle_dynamic_features(self):
        norm_current_coords = [self.norm_coords[i, self.current_location_idx[i]] for i in range(self.batch_size)]
        norm_current_load = [self.current_load[i][:, np.newaxis]/self.max_capacity[i] for i in range(self.batch_size)]
        norm_current_time = [self.current_time[i][:, np.newaxis]/self.planning_time[i] for i in range(self.batch_size)]
        features = [np.concatenate([norm_current_coords[i], norm_current_load[i], norm_current_time[i]], axis=1) for i in range(self.batch_size)]
        return features
    

    def get_travel_time(self):
        distances_list = [self.distance_matrix[i, self.current_location_idx[i], :] for i in range(self.batch_size)]
        distances_list = np.concatenate(distances_list).flatten()
        current_time_list = [np.repeat(self.current_time[i][:, np.newaxis], self.num_nodes, axis=1) for i in range(self.batch_size)]
        current_time_list = np.concatenate(current_time_list).flatten()
        planning_time_list = [np.asanyarray([self.planning_time[i]]*self.num_vehicles[i]) for i in range(self.batch_size)]
        time_horizon_list = [planning_time_list[i][:,np.newaxis]*TIME_HORIZONS for i in range(self.batch_size)]
        time_horizon_list = [np.repeat(time_horizon_list[i][:,np.newaxis,:], self.num_nodes, axis=1) for i in range(self.batch_size)]
        time_horizon_list = np.concatenate(time_horizon_list).reshape(-1, 6)
        road_types_list = [self.road_types[i, self.current_location_idx[i], :] for i in range(self.batch_size)]
        speed_profile_list = [SPEED_PROFILES[road_types_list[i], :] for i in range(self.batch_size)]
        speed_profile_list = np.concatenate(speed_profile_list).reshape(-1, 5)
        travel_time_list = compute_travel_time_vectorized(distances_list, current_time_list, time_horizon_list, speed_profile_list)
        travel_time_list = [travel_time_list[self.num_vehicles_cum[i-1]*self.num_nodes:self.num_vehicles_cum[i]*self.num_nodes] for i in range(1,self.batch_size+1)]
        travel_time_list = [travel_time_list[i].reshape(self.num_vehicles[i], -1) for i in range(self.batch_size)]
        return travel_time_list
        
    """
        travel_time
    """
    @property
    def node_dynamic_features(self):
        travel_time_list = self.get_travel_time()
        norm_travel_time_list = [travel_time_list[i]/self.planning_time[i] for i in range(self.batch_size)]
        return norm_travel_time_list

    @property
    def feasibility_mask(self):
        mask = [np.zeros((self.num_vehicles[i], self.num_nodes), dtype=bool) for i in range(self.batch_size)]
        is_pickup_visited = self.is_node_visited[:,1:self.num_requests+1]
        is_delivery_visited = self.is_node_visited[:,self.num_requests+1:]
        # for pickup, feasible if taken do not make load exceed capacity and not visited yet
        pickup_demands = self.demands[:,1:self.num_requests+1]
        current_load_if_pickup = [self.current_load[i][:, np.newaxis] + pickup_demands[i,:] for i in range(self.batch_size) ]
        is_pickup_exceed_load = [current_load_if_pickup[i]>self.max_capacity[i] for i in range(self.batch_size)]
        is_pickup_feasible = [np.logical_and(np.logical_not(is_pickup_visited[np.newaxis,i,:]), np.logical_not(is_pickup_exceed_load[i])) for i in range(self.batch_size)]
        # for delivery, feasible for the k-th vehicle 
        # if pickup is visited, and is assigned to the k-th vehicle
        # and it is not visited yet
        is_assigned_to_vec = [self.request_assignment[i][np.newaxis, :] == np.arange(self.num_vehicles[i])[:, np.newaxis] for i in range(self.batch_size)]
        is_delivery_feasible = [np.logical_and(is_pickup_visited[np.newaxis, i], np.logical_not(is_delivery_visited[np.newaxis, i])) for i in range(self.batch_size)]
        is_delivery_feasible = [np.repeat(is_delivery_feasible[i],self.num_vehicles[i], axis=0) for i in range(self.batch_size)]
        is_delivery_feasible = [np.logical_and(is_delivery_feasible[i], is_assigned_to_vec[i]) for i in range(self.batch_size)]
        for i in range(self.batch_size):
            mask[i][:,1:self.num_requests+1] = is_pickup_feasible[i]
            mask[i][:,self.num_requests+1:] = is_delivery_feasible[i]
        return mask

