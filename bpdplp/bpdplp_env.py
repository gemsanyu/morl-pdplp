import numpy as np

from bpdplp.bpdplp import TIME_HORIZONS, SPEED_PROFILES

def compute_travel_time(from_idx, to_idx, current_time, planning_time, distance_matrix, road_types):
    time_horizons = TIME_HORIZONS*planning_time
    speed_profile = SPEED_PROFILES[road_types[from_idx, to_idx]]
    distance = distance_matrix[from_idx, to_idx]
    
    horizon = np.searchsorted(time_horizons, current_time) - 1
    # horizon = 0
    # while time_horizons[horizon+1]<current_time:
    #     horizon += 1
    temp_time = current_time 
    while distance > 0:
        arrived_time = temp_time + distance/speed_profile[horizon]
        if arrived_time > time_horizons[horizon+1]:
            distance -= speed_profile[horizon]*(time_horizons[horizon+1]-temp_time)
            temp_time = time_horizons[horizon+1]
            horizon+=1
        else:
            distance=0
    travel_time = arrived_time-current_time
    print(arrived_time)
    exit()
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
        current_coords = [self.coords[i, self.current_location_idx[i]] for i in range(self.batch_size)]
        current_load = [self.current_load[i][:, np.newaxis] for i in range(self.batch_size)]
        current_time = [self.current_time[i][:, np.newaxis] for i in range(self.batch_size)]
        features = [np.concatenate([current_coords[i], current_load[i], current_time[i]], axis=1) for i in range(self.batch_size)]
        return features
    

    """
        travel_time
    """
    @property
    def node_dynamic_features(self):
        i = 0
        compute_travel_time(0, 1, 90, self.planning_time[i], self.distance_matrix[i], self.road_types[i])
        

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
