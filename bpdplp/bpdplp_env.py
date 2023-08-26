import time

import numpy as np
import numba as nb

from bpdplp.bpdplp import TIME_HORIZONS, SPEED_PROFILES

@nb.jit(nb.int64[:](nb.float32[:,:],nb.float32[:]),nopython=True,cache=True,parallel=True)
def find_passed_hz(time_horizons, current_times):
    passed_hz = np.empty(len(current_times), dtype=np.int64)
    _, n_hz = time_horizons.shape
    l_ct = len(current_times)
    for i in nb.prange(l_ct):
        for j in range(n_hz):
            if time_horizons[i,j] > current_times[i]:
                passed_hz[i] = j
                break
    return passed_hz

@nb.jit(nb.float32(nb.float32, nb.float32, nb.int64, nb.float32[:], nb.float32[:]), cache=True, nopython=True)
def compute_travel_time(distance, current_time, horizon, time_horizon, speed_profile):
    temp_time = current_time
    arrived_time = temp_time
    n_hz = len(time_horizon)
    for h in range(horizon, n_hz): 
    # while distance > 0:
        arrived_time = temp_time + distance/speed_profile[h]
        if arrived_time > time_horizon[h+1]:
            distance -= speed_profile[h]*(time_horizon[h+1]-temp_time)
            temp_time = time_horizon[h+1]
            # horizon+=1
        else:
            # distances[i]=0
            break
    travel_time = arrived_time-current_time
    return travel_time

@nb.jit(nb.float32[:](nb.float32[:], nb.float32[:],  nb.float32[:,:], nb.float32[:,:]), cache=True, nopython=True, parallel=True)
def compute_travel_time_loop(distances, current_times, time_horizons, speed_profiles):
    travel_times = np.empty(len(current_times), dtype=np.float32)
    horizons = find_passed_hz(time_horizons, current_times) - 1
    for i in nb.prange(len(current_times)):
        travel_times[i] = compute_travel_time(distances[i],current_times[i], horizons[i],time_horizons[i,:],speed_profiles[i,:])
    return travel_times    

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
        self.distance_matrix = distance_matrix.numpy().astype(np.float32)
        self.norm_distance_matrix = norm_distance_matrix.numpy()
        self.road_types = road_types.numpy()
        
        #for travel time computation
        planning_time_repeated = self.planning_time[:,np.newaxis,np.newaxis]
        planning_time_repeated = np.repeat(planning_time_repeated, self.max_num_vehicles, 1)
        planning_time_repeated = np.repeat(planning_time_repeated, self.num_nodes, 2)
        planning_time_repeated = planning_time_repeated.ravel()
        self.time_horizons_repeated = TIME_HORIZONS*planning_time_repeated[:, np.newaxis]
        #repeat-use variables
        self.is_depot_feasible = np.asanyarray([[[False]]*self.max_num_vehicles]*self.batch_size)
        self.vehicle_idx = np.arange(self.max_num_vehicles)[np.newaxis,:,np.newaxis]
        
        self.reset()
        

    def reset(self):
        self.current_load = np.zeros((self.batch_size, self.max_num_vehicles), dtype=np.float32)
        self.current_time = np.zeros((self.batch_size, self.max_num_vehicles), dtype=np.float32)
        self.current_horizons = np.zeros_like(self.current_time, dtype=np.int64)
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
        features = np.zeros((self.batch_size, self.num_nodes, 4), dtype=np.float32)
        # features[:,:,:2] = self.norm_coords
        features[:,:,0] = self.norm_demands
        features[:,:,1] = self.norm_service_durations
        features[:,:,2:] = self.norm_time_windows
        return features
    
    """
        current_coords, current_load, current_time
    """
    @property
    # @profile
    def vehicle_dynamic_features(self):
        # norm_current_coords = self.norm_coords[self.batch_vec_idx, self.current_location_idx.ravel(),:]
        # norm_current_coords = norm_current_coords.reshape(self.batch_size, self.max_num_vehicles, 2)
        norm_current_load = (self.current_load/self.max_capacity[:, np.newaxis])[:,:, np.newaxis]
        norm_current_time = (self.current_time/self.planning_time[:, np.newaxis])[:,:, np.newaxis]
        features = np.concatenate([norm_current_load,norm_current_time], axis=-1)
        return features
    
    # @profile
    def get_travel_time(self):
        current_location_idx = self.current_location_idx.ravel()
        distances_list = self.distance_matrix[self.batch_vec_idx, current_location_idx,:].ravel()
        current_time_list = self.current_time[:,:,np.newaxis]
        current_time_list = np.repeat(current_time_list,self.num_nodes,axis=2).ravel()
        time_horizon_list = self.time_horizons_repeated
        road_types_list = self.road_types[self.batch_vec_idx, current_location_idx,:].ravel()
        speed_profile_list = np.take(SPEED_PROFILES,road_types_list,axis=0)
        
        # passed_hz = find_passed_hz(time_horizon_list, current_time_list) - 1
        # passed_hz_compact = self.current_horizons
        # passed_hz_compact =  np.repeat(passed_hz_compact[:,:,np.newaxis],self.num_nodes,axis=2).ravel()
        # print(passed_hz, passed_hz_compact)
        # assert np.allclose(passed_hz, passed_hz_compact)
        travel_time_list = compute_travel_time_loop(distances_list.copy(), current_time_list.copy(), time_horizon_list.astype(np.float32), speed_profile_list)
        # travel_time_list = compute_travel_time_vectorizedv2(distances_list, current_time_list, time_horizon_list, speed_profile_list)
        travel_time_list = travel_time_list.reshape((self.batch_size, self.max_num_vehicles, self.num_nodes))
        return travel_time_list
        
    """
        travel_time
    """
    @property
    # @profile    
    def node_dynamic_features(self):
        travel_time_list = self.travel_time_list
        norm_travel_time_list = travel_time_list/self.planning_time[:,np.newaxis,np.newaxis]
        norm_travel_time_list = norm_travel_time_list[:,:,:,np.newaxis]
        return norm_travel_time_list


    @property
    # @profile
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
        is_assigned_to_vec = self.request_assignment[:,np.newaxis,:] == self.vehicle_idx
        is_delivery_feasible = np.logical_and(is_assigned_to_vec, is_pickup_visited[:,np.newaxis,:])
        is_delivery_feasible = np.logical_and(is_delivery_feasible, np.logical_not(is_delivery_visited[:,np.newaxis,:]))
        mask = np.concatenate([self.is_depot_feasible, is_pickup_feasible, is_delivery_feasible], axis=2)
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
        reward = self.service_node_by_vec(batch_idx, selected_vecs, selected_nodes)
        return *self.get_state(), reward

    """
        needs to be updated: current location, current time, current load, request assignment
        solution: tour_list, departure time
        objective vector: distance travelled, late penalty
    """
    # @profile
    def service_node_by_vec(self, batch_idx, selected_vecs, selected_nodes):
        # assert (len(batch_idx) == self.batch_size)
        travel_time_list = self.travel_time_list
        travel_time_vecs = travel_time_list[batch_idx, selected_vecs, selected_nodes]
        f1 = travel_time_vecs
        self.is_node_visited[batch_idx, selected_nodes] = True
        # isnp -> is_selected_node_pickup
        # assign the request to the vehicles
        isnp = selected_nodes <= self.num_requests
        # not_served = self.request_assignment[batch_idx[isnp], selected_nodes[isnp]-1] == -1
        # assert np.all(not_served)
        self.request_assignment[batch_idx[isnp], selected_nodes[isnp]-1] = selected_vecs[isnp]
        #add demands
        # isnd = selected_nodes > self.num_requests
        # selected_requests = selected_nodes.copy()
        # selected_requests[isnd] -= self.num_requests
        # assert np.all(self.request_assignment[batch_idx, selected_requests-1] == selected_vecs)
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
        f2 = None
        selected_vecs_current_time = self.current_time[batch_idx, selected_vecs]
        selected_nodes_tw = self.time_windows[batch_idx, selected_nodes]
        is_too_early = selected_vecs_current_time <= selected_nodes_tw[:,0]
        # if np.any(is_too_early):
        self.current_time[batch_idx[is_too_early], selected_vecs[is_too_early]] = selected_nodes_tw[is_too_early,0]
        is_too_late = selected_vecs_current_time > selected_nodes_tw[:,1]
        # if np.any(is_too_late):
        late_penalty = (selected_vecs_current_time[is_too_late]-selected_nodes_tw[is_too_late,1])
        if len(late_penalty)>0:
            self.late_penalty[batch_idx[is_too_late], selected_vecs[is_too_late]] += late_penalty
            # f2[is_too_late] = late_penalty
            f2 = np.empty_like(f1)
            f2[is_too_late] = late_penalty
            f2[np.logical_not(is_too_late)] = 0
        if f2 is None:
            f2 = np.zeros_like(f1)
            # exit()
        self.arrived_time[batch_idx, selected_vecs, self.num_visited_nodes[batch_idx, selected_vecs]] = self.current_time[batch_idx, selected_vecs]
        self.travel_cost[batch_idx, selected_vecs] += travel_time_vecs 
        self.current_location_idx[batch_idx, selected_vecs] = selected_nodes
        # after arriving, and start service, add service time to current time
        self.current_time[batch_idx, selected_vecs] += self.service_durations[batch_idx, selected_nodes]
        # self.update_curr_horizons(batch_idx, selected_vecs)
        self.travel_time_list = self.get_travel_time()
        if self.num_visited_nodes[0,0] == 2*self.num_requests: #add travel time to depots
            repeated_vec_idx = np.arange(self.max_num_vehicles)[np.newaxis, :]
            repeated_vec_idx = np.broadcast_to(repeated_vec_idx[0,:],(self.batch_size,self.max_num_vehicles)).ravel()
            travel_time_to_depot = + self.travel_time_list[self.batch_vec_idx, repeated_vec_idx, 0].reshape(self.batch_size,self.max_num_vehicles)
            f1 = f1 + travel_time_to_depot.sum(axis=1)
        return np.concatenate([f1[:, np.newaxis], f2[:, np.newaxis]], axis=-1)

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
        repeated_vec_idx = np.broadcast_to(repeated_vec_idx[0,:],(self.batch_size,self.max_num_vehicles)).ravel()
        # travel_time = self.get_travel_time()
        travel_time = self.travel_time_list
        self.travel_cost[self.batch_vec_idx, repeated_vec_idx] += travel_time[self.batch_vec_idx, repeated_vec_idx, 0]
        # for i in range(self.batch_size):
        #     for k in range(self.num_vehicles[i]):
        #         self.travel_cost[i][k] += travel_time[i][k,0]
        # #batch-wise travel cost and penalty
        # assert (self.current_load.sum() == 0)
        travel_cost = self.travel_cost.sum(axis=1)
        late_penalty = self.late_penalty.sum(axis=1)
        return self.tour_list, self.arrived_time, self.departure_time_list, travel_cost, late_penalty