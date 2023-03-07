import time

import numpy as np

from bpdplp.bpdplp import TIME_HORIZONS, SPEED_PROFILES

def compute_travel_time_vectorized(distances, current_times, time_horizons, speed_profiles):
    distances = distances.copy()
    num_pair = distances.shape[0]
    pair_idx = np.arange(num_pair)
    horizons = np.argmax(time_horizons > current_times[:, np.newaxis], axis=1)-1
    temp_times = current_times.copy()
    is_distance_nonzero = distances > 0
    while np.any(is_distance_nonzero):
        arrived_time = temp_times + distances/speed_profiles[pair_idx, horizons]
        is_arrived_time_pass_breakpoint = arrived_time > time_horizons[pair_idx, horizons+1]
        #is_nonzero_and_not_pass_breakpoint
        inanpb = np.logical_and(is_distance_nonzero, np.logical_not(is_arrived_time_pass_breakpoint))
        temp_times[inanpb] = arrived_time[inanpb]
        distances[inanpb] = 0
        #is_nonzero_and_pass_breakpoint
        inapb = np.logical_and(is_distance_nonzero, is_arrived_time_pass_breakpoint)
        distances[inapb] -= speed_profiles[pair_idx[inapb], horizons[inapb]]*(time_horizons[pair_idx[inapb],horizons[inapb]+1]-temp_times[inapb])
        temp_times[inapb] = time_horizons[pair_idx[inapb],horizons[inapb]+1]
        horizons[inapb] = horizons[inapb] + 1
        is_distance_nonzero = distances > 0
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

"""
    we need to add dummy vehicles,
    so that all computation here is vectorized
    this is really bottleneck
"""
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
        self.max_num_vehicles = int(np.max(self.num_vehicles))
        self.total_num_vehicles = int(self.num_vehicles_cum[-1])
        self.max_capacity = max_capacity.numpy()
        self.batch_size, self.num_nodes, _ = coords.shape
        self.batch_vec_idx = np.repeat(np.arange(self.batch_size), self.max_num_vehicles)
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
        
        #for travel time computation
        planning_time_repeated = self.planning_time[:,np.newaxis,np.newaxis]
        planning_time_repeated = np.repeat(planning_time_repeated, self.max_num_vehicles, 1)
        planning_time_repeated = np.repeat(planning_time_repeated, self.num_nodes, 2)
        planning_time_repeated = planning_time_repeated.flatten()
        self.time_horizons_repeated = TIME_HORIZONS*planning_time_repeated[:, np.newaxis]
        self.reset()

    def reset(self):
        self.current_load = np.zeros((self.batch_size, self.max_num_vehicles), dtype=np.float32)
        self.current_time = np.zeros((self.batch_size, self.max_num_vehicles), dtype=np.float32)
        self.current_location_idx = np.zeros((self.batch_size, self.max_num_vehicles), dtype=int)
        self.is_not_dummy_mask = np.ones((self.batch_size, self.max_num_vehicles, self.num_nodes), dtype=bool)
        for i in range(self.batch_size):
            self.is_not_dummy_mask[i,self.num_vehicles[i]:,:] = False
        self.is_node_visited = np.zeros((self.batch_size, self.num_nodes), dtype=bool)
        self.is_node_visited[:, 0] = True
        self.request_assignment = np.full((self.batch_size, self.num_requests), -1, dtype=int)
        self.travel_cost = np.zeros((self.batch_size, self.max_num_vehicles), dtype=np.float32)
        self.late_penalty = np.zeros((self.batch_size, self.max_num_vehicles), dtype=np.float32)
        self.num_visited_nodes = np.zeros((self.batch_size, self.max_num_vehicles), dtype=int)
        self.arrived_time = np.zeros((self.batch_size, self.max_num_vehicles, self.num_nodes), dtype=np.float32)
        self.tour_list = np.zeros((self.batch_size, self.max_num_vehicles, self.num_nodes), dtype=int)
        self.departure_time_list = np.zeros((self.batch_size, self.max_num_vehicles, self.num_nodes), dtype=np.float32)
        self.travel_time_list = self.get_travel_time()
        

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
        norm_current_coords = self.norm_coords[self.batch_vec_idx, self.current_location_idx.flatten(),:]
        norm_current_coords = norm_current_coords.reshape(self.batch_size, self.max_num_vehicles, 2)
        norm_current_load = (self.current_load/self.max_capacity[:, np.newaxis])[:,:, np.newaxis]
        norm_current_time = (self.current_time/self.planning_time[:, np.newaxis])[:,:, np.newaxis]
        features = np.concatenate([norm_current_coords,norm_current_load,norm_current_time], axis=-1)
        return features
    

    def get_travel_time(self):
        current_location_idx = self.current_location_idx.flatten()
        distances_list = self.distance_matrix[self.batch_vec_idx, current_location_idx,:].flatten()
        current_time_list = self.current_time[:,:,np.newaxis]
        current_time_list = np.repeat(current_time_list,self.num_nodes,axis=2).flatten()
        time_horizon_list = self.time_horizons_repeated
        road_types_list = self.road_types[self.batch_vec_idx, current_location_idx,:].flatten()
        speed_profile_list = SPEED_PROFILES[road_types_list,:]
        travel_time_list = compute_travel_time_vectorized(distances_list, current_time_list, time_horizon_list, speed_profile_list)
        travel_time_list = travel_time_list.reshape((self.batch_size, self.max_num_vehicles, self.num_nodes))
        # travel_time_list = [travel_time_list[self.num_vehicles_cum[i-1]*self.num_nodes:self.num_vehicles_cum[i]*self.num_nodes] for i in range(1,self.batch_size+1)]
        # travel_time_list = [travel_time_list[i].reshape(self.num_vehicles[i], -1) for i in range(self.batch_size)]
        
        # print("-----------------------------")
        # distances_listv2 = [self.distance_matrix[i, self.current_location_idx[i], :] for i in range(self.batch_size)]
        # current_time_listv2 = [self.current_time[i] for i in range(self.batch_size)]
        # planning_time_listv2 = [np.asanyarray([self.planning_time[i]]*self.num_vehicles[i]) for i in range(self.batch_size)]
        # time_horizon_listv2 = [planning_time_listv2[i][:,np.newaxis]*TIME_HORIZONS for i in range(self.batch_size)]
        # road_types_listv2 = [self.road_types[i, self.current_location_idx[i], :] for i in range(self.batch_size)]
        # speed_profile_listv2 = [SPEED_PROFILES[road_types_listv2[i], :] for i in range(self.batch_size)]
        # travel_time_listv2 = []
        # print(distances_list) 
        # print(np.concatenate(distances_listv2).flatten())
        
        # print("+++++++++++++++++++")
        # z = 0
        # for i in range(self.batch_size):
        #     travel_time_batch = []
        #     for k in range(self.num_vehicles[i]):
        #         travel_time_vec = []
        #         for j in range(self.num_nodes):
        #             distance = distances_listv2[i][k,j]
        #             current_time = current_time_listv2[i][k]
        #             time_horizon = time_horizon_listv2[i][k]
        #             speed_profile = speed_profile_listv2[i][k,j]
        #             z+=1
        #             travel_time = compute_travel_time(distance, current_time, time_horizon, speed_profile)
        #             travel_time_vec += [travel_time]
        #         travel_time_batch += [travel_time_vec]
        #     travel_time_listv2 += [np.asanyarray(travel_time_batch)]
        # print(travel_time_list)
        # print(travel_time_listv2)
        # print("-----------------------------")
        # for i in range(self.batch_size):
        #     #assert np.all(np.isclose(travel_time_list[i], travel_time_listv2[i]))
        #     # print(np.isclose(travel_time_list[i], travel_time_listv2[i]))
        return travel_time_list
        
    """
        travel_time
    """
    @property
    def node_dynamic_features(self):
        travel_time_list = self.travel_time_list
        norm_travel_time_list = travel_time_list/self.planning_time[:,np.newaxis,np.newaxis]
        norm_travel_time_list = norm_travel_time_list[:,:,:,np.newaxis]
        return norm_travel_time_list

    @property
    def feasibility_mask(self):
        is_pickup_visited = self.is_node_visited[:,1:self.num_requests+1]
        is_delivery_visited = self.is_node_visited[:,self.num_requests+1:]
        # for pickup, feasible if taken do not make load exceed capacity and not visited yet
        pickup_demands = self.demands[:,np.newaxis,1:self.num_requests+1]
        current_load_if_pickup = self.current_load[:,:,np.newaxis] + pickup_demands
        is_pickup_exceed_load = current_load_if_pickup > self.max_capacity[:,np.newaxis,np.newaxis]
        is_pickup_feasible = np.logical_and(np.logical_not(is_pickup_visited[:,np.newaxis,:]), np.logical_not(is_pickup_exceed_load))
        # for delivery, feasible for the k-th vehicle 
        # if pickup is visited, and is assigned to the k-th vehicle
        # and it is not visited yet
        # is_assigned_to_vec = 
        vehicle_idx = np.arange(self.max_num_vehicles)[np.newaxis,:,np.newaxis]
        is_assigned_to_vec = self.request_assignment[:,np.newaxis,:] == vehicle_idx
        is_delivery_feasible = np.logical_and(is_assigned_to_vec, is_pickup_visited[:,np.newaxis,:])
        is_delivery_feasible = np.logical_and(is_delivery_feasible, np.logical_not(is_delivery_visited[:,np.newaxis,:]))
        is_depot_feasible = np.asanyarray([[[False]]*self.max_num_vehicles]*self.batch_size)
        mask = np.concatenate([is_depot_feasible, is_pickup_feasible, is_delivery_feasible], axis=2)
        # lastly mask the dummy vehicles
        mask = np.logical_and(mask, self.is_not_dummy_mask)
        return mask
    """
        i think we need to vectorize this, because a lot of it can be...
        what cannot be vectorized:
        1. tour list,
        2. departure time list,
        3. current time,
        4. arrived time,
        5. travel cost,
        6. current location,
    """
    def act(self, batch_idx, selected_vecs, selected_nodes):
        #just send the vehicle to the node
        # selected_nodes = np.asanyarray(selected_nodes)
        # selected_vecs = np.asanyarray(selected_vecs)
        self.service_node_by_vec(batch_idx, selected_vecs, selected_nodes)


    """
        needs to be updated: current location, current time, current load, request assignment
        solution: tour_list, departure time
        objective vector: distance travelled, late penalty
    """
    def service_node_by_vec(self, batch_idx, selected_vecs, selected_nodes):
        travel_time_list = self.travel_time_list
        travel_time_vecs = travel_time_list[batch_idx, selected_vecs, selected_nodes]
        self.is_node_visited[batch_idx, selected_nodes] = True
        # isnp -> is_selected_node_pickup
        # assign the request to the vehicles
        isnp = selected_nodes <= self.num_requests
        self.request_assignment[batch_idx[isnp], selected_nodes[isnp]-1] = selected_vecs[isnp]
        #add demands
        selected_nodes_demands = self.demands[batch_idx, selected_nodes]
        self.current_load[batch_idx, selected_vecs] += selected_nodes_demands
        self.num_visited_nodes[batch_idx, selected_vecs] += 1
        self.tour_list[batch_idx, selected_vecs, self.num_visited_nodes[batch_idx, selected_vecs]] = selected_nodes
        self.departure_time_list[batch_idx, selected_vecs, self.num_visited_nodes[batch_idx, selected_vecs]] = self.current_time[batch_idx, selected_vecs]
        #add to travel time to current time
        self.current_time[batch_idx, selected_vecs] += travel_time_vecs

        # now filter the actions or the selected vehicles based on their current time
        # if early, then ceil,
        # if late, then add to penalty  
        selected_vecs_current_time = self.current_time[batch_idx, selected_vecs]
        selected_nodes_tw = self.time_windows[batch_idx, selected_nodes]
        is_too_early = selected_vecs_current_time <= selected_nodes_tw[:,0]
        if np.any(is_too_early):
            self.current_time[batch_idx[is_too_early], selected_vecs[is_too_early]] = selected_nodes_tw[is_too_early,0]
        is_too_late = selected_vecs_current_time > selected_nodes_tw[:,1]
        if np.any(is_too_late):
            self.late_penalty[batch_idx[is_too_late], selected_vecs[is_too_late]] += (selected_vecs_current_time[is_too_late]-selected_nodes_tw[is_too_late,1])
        
        self.arrived_time[batch_idx, selected_vecs, self.num_visited_nodes[batch_idx, selected_vecs]] = self.current_time[batch_idx, selected_vecs]
        self.travel_cost[batch_idx, selected_vecs] += travel_time_vecs 
        self.current_location_idx[batch_idx, selected_vecs] = selected_nodes
        # after arriving, and start service, add service time to current time
        self.current_time[batch_idx, selected_vecs] += self.service_durations[batch_idx, selected_nodes]
        self.travel_time_list = self.get_travel_time()

    def get_state(self):
        return self.vehicle_dynamic_features, self.node_dynamic_features, self.feasibility_mask 

    """
        should we return the batch-wise results already here?
        i think so
        NOTES:
        previously, the time travelled from the last location visited back
        to the depot is not added, now time to add it
        i think that's it,,
        the late penalty back to depot is not considered, or
        will it be considered? 
    """
    def finish(self):
        repeated_vec_idx = np.arange(self.max_num_vehicles)[np.newaxis, :]
        repeated_vec_idx = np.repeat(repeated_vec_idx, self.batch_size, axis=0).flatten()
        travel_time = self.get_travel_time()
        self.travel_cost[self.batch_vec_idx, repeated_vec_idx] += travel_time[self.batch_vec_idx, repeated_vec_idx, 0]
        # for i in range(self.batch_size):
        #     for k in range(self.num_vehicles[i]):
        #         self.travel_cost[i][k] += travel_time[i][k,0]
        #batch-wise travel cost and penalty
        travel_cost = self.travel_cost.sum(axis=1)
        late_penalty = self.late_penalty.sum(axis=1)
        return self.tour_list, self.arrived_time, self.departure_time_list, travel_cost, late_penalty