import numpy as np

"""
    we need to add dummy vehicles,
    so that all computation here is vectorized
    this is really bottleneck
"""
class PDPTW_Env(object):
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
                 norm_distance_matrix) -> None:
        
        self.num_vehicles = num_vehicles.numpy()
        self.num_vehicles_cum = np.insert(np.cumsum(self.num_vehicles),0,0)
        self.max_num_vehicles = int(np.max(self.num_vehicles))
        self.total_num_vehicles = int(self.num_vehicles_cum[-1])
        self.max_capacity = max_capacity.numpy()
        self.batch_size, self.num_nodes, _ = coords.shape
        self.batch_vec_idx = np.repeat(np.arange(self.batch_size), self.max_num_vehicles)
        self.batch_vec_idx_not_flat = np.repeat(np.arange(self.batch_size)[:,np.newaxis], self.max_num_vehicles, axis=1)
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
        self.num_visited_nodes = np.zeros((self.batch_size, self.max_num_vehicles), dtype=int)
        self.arrived_time = np.zeros((self.batch_size, self.max_num_vehicles, self.num_nodes), dtype=np.float32)
        self.tour_list = [[[0] for _ in range(self.max_num_vehicles)] for _ in range(self.batch_size)]
        self.departure_time_list = np.zeros((self.batch_size, self.max_num_vehicles, self.num_nodes), dtype=np.float32)
        

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
        travel_time_list = self.distance_matrix[self.batch_vec_idx, current_location_idx,:].reshape((self.batch_size, self.max_num_vehicles, self.num_nodes))
        return travel_time_list
        
    """
        travel_time
    """
    @property
    # @profile    
    def node_dynamic_features(self):
        node_dynamic_features = []
        travel_time_list = self.get_travel_time()
        norm_travel_time_list = travel_time_list/self.planning_time[:,np.newaxis,np.newaxis]
        norm_travel_time_list = norm_travel_time_list[:,:,:,np.newaxis]
        node_dynamic_features += [norm_travel_time_list]
        
        # current_time = self.current_time[:,:,np.newaxis]
        # arrival_time = travel_time_list + current_time
        # late_tw = self.time_windows[:,:,1][:,np.newaxis,:]
        # diff_arrival_time_to_late_tw = late_tw - arrival_time
        # norm_diff_at_to_ltw = diff_arrival_time_to_late_tw/self.planning_time[:,np.newaxis,np.newaxis]
        # norm_diff_at_to_ltw = norm_diff_at_to_ltw[:,:,:,np.newaxis]
        # node_dynamic_features += [norm_diff_at_to_ltw]

        late_tw = self.time_windows[:,:,1][:,np.newaxis,:]
        current_time = self.current_time[:,:,np.newaxis]
        diff_from_current_time_to_late_tw = late_tw-current_time
        norm_dfc_to_ltw = diff_from_current_time_to_late_tw/self.planning_time[:,np.newaxis,np.newaxis]
        norm_dfc_to_ltw = norm_dfc_to_ltw[:,:,:,np.newaxis]
        norm_dfc_to_ltw[norm_dfc_to_ltw<0] =0
        node_dynamic_features += [norm_dfc_to_ltw]
        
        node_dynamic_features = np.concatenate(node_dynamic_features,axis=-1)
        return node_dynamic_features


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
        # check if violate time window
        travel_time_list = self.get_travel_time()
        arrival_time = self.current_time[:,:,np.newaxis] + travel_time_list
        late_time_window = self.time_windows[:,:,1][:,np.newaxis,:]
        will_arrive_late = arrival_time > late_time_window
        mask = np.logical_and(mask, np.logical_not(will_arrive_late))
        # now check if distance to from each vehicle's current location
        # to the node and back to depot is still less than planning time (H)
        time_to_depot_from_all = self.distance_matrix[:,:,0][:,np.newaxis,:]
        early_time_window = self.time_windows[:,:,0][:,np.newaxis,:]
        start_service_time = np.maximum(arrival_time, early_time_window)
        finish_service_time = start_service_time + self.service_durations[:, np.newaxis, :]
        time_arrive_back_to_depot = finish_service_time + time_to_depot_from_all
        not_late_back_to_depot = time_arrive_back_to_depot < self.planning_time[:, np.newaxis, np.newaxis]
        mask = np.logical_and(mask, not_late_back_to_depot)        
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
        return self.get_state()

    """
        needs to be updated: current location, current time, current load, request assignment
        solution: tour_list, departure time
        objective vector: distance travelled, late penalty
    """
    # @profile
    def service_node_by_vec(self, batch_idx, selected_vecs, selected_nodes):
        # assert (len(batch_idx) == self.batch_size)
        travel_time_list = self.get_travel_time()
        travel_time_vecs = travel_time_list[batch_idx, selected_vecs, selected_nodes]
        self.is_node_visited[batch_idx, selected_nodes] = True
        # isnp -> is_selected_node_pickup
        # assign the request to the vehicles
        isnp = selected_nodes <= self.num_requests
        # not_served = self.request_assignment[batch_idx[isnp], selected_nodes[isnp]-1] == -1
        # assert np.all(not_served)
        self.request_assignment[batch_idx[isnp], selected_nodes[isnp]-1] = selected_vecs[isnp]
        #add demands
        selected_nodes_demands = self.demands[batch_idx, selected_nodes]
        self.current_load[batch_idx, selected_vecs] += selected_nodes_demands
        self.num_visited_nodes[batch_idx, selected_vecs] += 1
        for i, idx in enumerate(batch_idx):
            vec = selected_vecs[i]
            node = selected_nodes[i]
            self.tour_list[idx][vec] += [node] 
        self.departure_time_list[batch_idx, selected_vecs, self.num_visited_nodes[batch_idx, selected_vecs]] = self.current_time[batch_idx, selected_vecs]
        #add to travel time to current time
        self.current_time[batch_idx, selected_vecs] += travel_time_vecs
        
        # now filter the actions or the selected vehicles based on their current time
        # if early, then ceil,
        # if late, then add to penalty
        selected_vecs_current_time = self.current_time[batch_idx, selected_vecs]
        selected_nodes_tw = self.time_windows[batch_idx, selected_nodes]
        is_too_early = selected_vecs_current_time <= selected_nodes_tw[:,0]
        # if np.any(is_too_early):
        self.current_time[batch_idx[is_too_early], selected_vecs[is_too_early]] = selected_nodes_tw[is_too_early,0]
        self.arrived_time[batch_idx, selected_vecs, self.num_visited_nodes[batch_idx, selected_vecs]] = self.current_time[batch_idx, selected_vecs]
        self.current_location_idx[batch_idx, selected_vecs] = selected_nodes
        self.current_time[batch_idx, selected_vecs] += self.service_durations[batch_idx, selected_nodes]
        
    def get_state(self):
        return self.vehicle_dynamic_features, self.node_dynamic_features, self.feasibility_mask 

    """
        the chosen tour list will now be evaluated
        and invalid nodes will be removed?
        from the action sequence, invalid nodes will
        be: deliver request served, but its delivery is not.
        it will make that request invalid.
    """
    def finish(self):
        max_visited_nodes = 0
        for i in range(self.batch_size):
            for k in range(self.max_num_vehicles):
                real_tour = []
                for l in range(len(self.tour_list[i][k])):
                    # check if pickup, if it is pickup,
                    # then check if its corresponding delivery is visited too
                    node = self.tour_list[i][k][l]
                    is_valid = True
                    if node > 0 and node <= self.num_requests:
                        delivery_node = node + self.num_requests
                        if not delivery_node in self.tour_list[i][k]:
                            is_valid = False
                            self.is_node_visited[i,node]=False
                    if is_valid:
                        real_tour += [node]
                self.tour_list[i][k] = real_tour
                max_visited_nodes = max(len(self.tour_list[i][k]), max_visited_nodes)
        
        # pad so all of them have same tour count
        for i in range(self.batch_size):
            for k in range(self.max_num_vehicles):
                len_diff = max_visited_nodes - len(self.tour_list[i][k])
                self.tour_list[i][k] += [0]*len_diff
        self.tour_list = np.asanyarray(self.tour_list, dtype=int)
        
        # now compute travel time
        A = self.tour_list
        B = np.roll(A, shift=-1, axis=-1)
        travel_time_ = self.distance_matrix[:, A, B]
        batch_idx = np.arange(self.batch_size, dtype=int)
        travel_time = travel_time_[batch_idx, batch_idx, :, :]
        total_travel_time = np.sum(travel_time.reshape(self.batch_size, -1), axis=-1)
        
        # we need to assert 
        # 1. capacity constraints
        # tour_ = self.tour_list.reshape(self.batch_size,-1)
        # _, len_tour_ = tour_.shape
        # tour_vec_idx = np.arange(self.batch_size)[:, np.newaxis]
        # tour_vec_idx = np.repeat(tour_vec_idx, len_tour_, axis=-1)
        # tour_load = self.demands[tour_vec_idx, tour_].reshape(self.batch_size, self.max_num_vehicles, -1)
        # tour_load = np.cumsum(tour_load, axis=-1)
        # is_capacity_respected = np.logical_and(tour_load<=self.max_capacity[:,np.newaxis,np.newaxis], tour_load>=0)
        # assert np.all(is_capacity_respected)
        # # 2. precedence constraints
        # for i in range(self.batch_size):
        #     for k in range(self.max_num_vehicles):
        #         tour = self.tour_list[i,k,:]
        #         for l in range(len(tour)):
        #             node = tour[l]
        #             if node > self.num_requests:
        #                 pickup_node = node-self.num_requests
        #                 # print(node, tour[:l], tour)
        #                 assert pickup_node in tour[:l]
        # # 3. time window constraints and planning time constraints
        # for i in range(self.batch_size):
        #     for k in range(self.max_num_vehicles):
        #         current_time = 0
        #         tour = self.tour_list[i,k,:]
        #         for l in range(1,len(tour)):
        #             node = tour[l]
        #             prev_node = tour[l-1]
        #             tt = self.distance_matrix[i,prev_node,node]
        #             current_time = current_time + tt
        #             current_time = max(current_time, self.time_windows[i,node,0])
        #             assert current_time <= self.time_windows[i,node,1]
        #             current_time = current_time + self.service_durations[i,node]
        #         # 4. planning time constraints.
        #         t_to_depot = self.distance_matrix[i,node,0]
        #         current_time = current_time + t_to_depot
        #         assert current_time <= self.planning_time[i]
        # assertions end
        
        # compute not served penalty
        num_node_not_visited = np.sum(np.logical_not(self.is_node_visited), axis=-1)
        return self.tour_list, self.arrived_time, self.departure_time_list, total_travel_time, num_node_not_visited