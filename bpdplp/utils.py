import os
import pathlib

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import haversine_distances

# road types
SLOW=0
NORMAL=1
FAST=2

#distribution of nodes
RANDOM=0
RANDOMCLUSTER=1
CLUSTER=2

#depot location choice
# RANDOM=0
CENTRAL=1

def read_instance(instance_path):
    with open(instance_path.absolute(), "r") as instance_file:
        lines = instance_file.readlines()
        for i in range(11):
            strings = lines[i].split()
            if i == 4:
                num_nodes = int(strings[1])
            elif i == 7:
                planning_time = int(strings[1])
            elif i == 9:
                max_capacity = float((strings[1]))
        coords = np.zeros((num_nodes,2), dtype=np.float32)
        demands = np.zeros((num_nodes,), dtype=np.float32)
        time_windows = np.zeros((num_nodes,2), dtype=np.float32)
        service_durations = np.zeros((num_nodes,), dtype=np.float32)
        for i in range(11, num_nodes+11):
            strings = lines[i].split()
            idx = i-11
            coords[idx,0], coords[idx,1] = float(strings[1]), float(strings[2])
            demands[idx] = float(strings[3])
            time_windows[idx,0], time_windows[idx,1] = strings[4], strings[5]
            service_durations[idx] = strings[6]
        distance_matrix = np.zeros((num_nodes,num_nodes), dtype=np.float32)
        for i in range(num_nodes+12, 2*num_nodes+12):
            strings = lines[i].split()
            idx = i-(num_nodes+12)
            for j in range(num_nodes):
                distance_matrix[idx,j] = float(strings[j])
    return num_nodes, planning_time, max_capacity, coords, demands, time_windows, service_durations, distance_matrix

def read_road_types(road_types_path, num_nodes):
    road_types = np.zeros((num_nodes,num_nodes), dtype=np.int8)
    if not os.path.isfile(road_types_path.absolute()):
        a = np.random.randint(0,3,size=(num_nodes, num_nodes), dtype=np.int8)
        road_types = np.tril(a) + np.tril(a, -1).T
        with open(road_types_path.absolute(), "w") as road_types_file:
            for i in range(num_nodes):
                line = ""
                for j in range(num_nodes):
                    line += str(road_types[i,j])+" "
                road_types_file.write(line+"\n")
    else:
        with open(road_types_path.absolute(), "r") as road_types_file:
            lines = road_types_file.readlines()
            for i,line in enumerate(lines):
                strings = line.split()
                for j in range(num_nodes):
                    road_types[i,j] = int(strings[j])
    return road_types

def read_graph(graph_seed):
    # try reading saved numpyz arrays,, it took long to read fromt xt
    graph_numpy_path = pathlib.Path(".")/"dataset"/"graphs"/(graph_seed+".npz")
    if os.path.isfile(graph_numpy_path.absolute()):
        data = np.load(graph_numpy_path.absolute())
        coords = data["coords"]
        distance_matrix = data["distance_matrix"]
        haversine_matrix = data["haversine_matrix"]
        return coords, distance_matrix, haversine_matrix

    graph_path = pathlib.Path(".")/"dataset"/"graphs"/graph_seed
    # graph
    num_nodes = 0
    coords = None
    distance_matrix = None
    with open(graph_path.absolute(), "r") as graph_file:
        lines = graph_file.readlines()
        num_nodes = int(lines[0].split()[1])
        coords = np.zeros((num_nodes,2), dtype=np.float32)
        for i in range(1, num_nodes+1):
            strings = lines[i].split()
            idx = i-1
            coords[idx, 0], coords[idx,1] = float(strings[1]), float(strings[2])
        
        distance_matrix = np.zeros((num_nodes,num_nodes), dtype=np.float32)
        for i in range(num_nodes+2,2*num_nodes+2):
            strings = lines[i].split()
            idx = i-(num_nodes+2)
            for j in range(num_nodes):
                distance_matrix[idx,j] = float(strings[j])
    haversine_matrix = haversine_distances(coords)*6371000/1000 # to get kilometer
    # save to numpy
    np.savez(graph_numpy_path.absolute(), coords=coords, distance_matrix=distance_matrix, haversine_matrix=haversine_matrix) 
    return coords, distance_matrix, haversine_matrix


def sample_cluster_from_graph(num_nodes, num_cluster, coords, haversine_matrix, cluster_delta=1):
    seed_locations_idx = np.random.choice(len(coords), num_cluster, replace=False)
    num_other_nodes = num_nodes-num_cluster
    distance_to_seed = haversine_matrix[seed_locations_idx, :]
    distance_to_seed[:, seed_locations_idx] = 1000000
    probs = np.exp(-distance_to_seed*cluster_delta).sum(axis=0)
    probs = probs/probs.sum()
    chosen_nodes_idx = np.random.choice(len(coords), (num_other_nodes), p=probs, replace=False)
    chosen_nodes_idx = np.concatenate([seed_locations_idx, chosen_nodes_idx], axis=0)
    chosen_nodes_coords = coords[chosen_nodes_idx]
    kmeans = KMeans(n_clusters=num_cluster).fit(chosen_nodes_coords)
    clusters = [chosen_nodes_idx[np.nonzero((kmeans.labels_==i))] for i in range(num_cluster)]
    
    return clusters


def generate_graph(graph_seed, num_nodes, num_cluster, cluster_delta, distribution, depot_location):
    coords_L, distance_matrix_L, haversine_matrix_L = graph_seed
    num_nodes_L = len(coords_L)
    chosen_nodes_idx = None
    if distribution == RANDOM:
        chosen_nodes_idx = np.random.choice(num_nodes_L, size=(num_nodes-1))
    elif distribution == RANDOMCLUSTER:
        h = np.random.random()*0.2+0.4
        num_clustered_nodes = int(h*(num_nodes-1))
        num_cluster = min(num_cluster, num_clustered_nodes)
        num_random_nodes = num_nodes-1-num_clustered_nodes
        clusters = sample_cluster_from_graph(num_clustered_nodes, num_cluster, coords_L, haversine_matrix_L, cluster_delta)
        cluster_nodes_idx = np.concatenate(clusters)
        remaining_nodes_idx = np.setdiff1d(np.arange(num_nodes_L), cluster_nodes_idx)
        random_nodes_idx = np.random.choice(remaining_nodes_idx, size=(num_random_nodes), replace=False)
        chosen_nodes_idx = np.concatenate([cluster_nodes_idx, random_nodes_idx], axis=0)
    else: #CLUSTER
        clusters = sample_cluster_from_graph(num_nodes-1, num_cluster, coords_L, haversine_matrix_L, cluster_delta)
        even_clusters = []
        odd_clusters = []
        for cluster in clusters:
            if len(cluster) %2 ==0:
                even_clusters += [cluster]
            else:
                odd_clusters += [cluster]
        while len(odd_clusters) > 0:
            # get biggest
            list_of_sizes = np.asanyarray([len(cluster) for cluster in odd_clusters])
            largest_cluster_idx = np.argmax(list_of_sizes)
            largest_cluster = odd_clusters[largest_cluster_idx]
            # remove biggest from remaining clusters
            odd_clusters = odd_clusters[:largest_cluster_idx] + odd_clusters[largest_cluster_idx+1:] 
            # get central of remaining clusters
            min_coords_clusters = [np.min(coords_L[cluster], axis=0, keepdims=True) for cluster in odd_clusters]
            max_coords_clusters = [np.max(coords_L[cluster], axis=0, keepdims=True) for cluster in odd_clusters]
            central_coords_clusters = [(min_coords_clusters[i]+max_coords_clusters[i])/2 for i in range(len(odd_clusters))]
            central_coords_clusters = np.concatenate(central_coords_clusters, axis=0)
            haversine_from_bigger_to_centrals = haversine_distances(coords_L[largest_cluster], central_coords_clusters)
            # get the node idx to move and move it
            idx_to_move = np.argmin(haversine_from_bigger_to_centrals)
            cluster_to_move_into_idx = int(idx_to_move/len(largest_cluster))
            node_idx_to_move_idx = idx_to_move%len(largest_cluster)
            node_idx_to_move = largest_cluster[node_idx_to_move_idx]
            #remove from largest cluster
            largest_cluster = np.delete(largest_cluster, node_idx_to_move_idx)
            #add to the cluster
            cluster_to_move_into = odd_clusters[cluster_to_move_into_idx]
            cluster_to_move_into = np.append(cluster_to_move_into, node_idx_to_move)
            #remove the appended cluster because it is even now
            odd_clusters = odd_clusters[:cluster_to_move_into_idx] + odd_clusters[cluster_to_move_into_idx+1:]
            #add those clusters into even clusters
            even_clusters += [largest_cluster, cluster_to_move_into]
        clusters = even_clusters
        # now for each cluster we have to split into two halves
        # first half pickup, second half delivery
        pickup_nodes_list = []
        delivery_nodes_list = []
        chosen_nodes_idx = []
        for cluster in clusters:
            len_cluster = len(cluster)
            pickup_nodes_list += [cluster[:int(len_cluster/2)]]
            delivery_nodes_list += [cluster[int(len_cluster/2):]]
        chosen_nodes_idx = np.concatenate(pickup_nodes_list+delivery_nodes_list, axis=0)

    remaining_nodes_idx =  np.setdiff1d(np.arange(num_nodes_L), chosen_nodes_idx)
    if depot_location == RANDOM:
        chosen_depot_idx = np.random.choice(remaining_nodes_idx, size=(1))
        chosen_nodes_idx = np.concatenate([chosen_depot_idx, chosen_nodes_idx], axis=0)
    else:
        chosen_nodes_coords = coords_L[chosen_nodes_idx]
        min_coords, max_coords = np.min(chosen_nodes_coords, axis=0, keepdims=True), np.max(chosen_nodes_coords, axis=0, keepdims=True)
        center_coords = (min_coords+max_coords)/2
        remaining_coords = coords_L[remaining_nodes_idx]
        distance_to_center = haversine_distances(remaining_coords, center_coords)
        chosen_depot_idx_ = np.argmin(distance_to_center, keepdims=True).squeeze(0)
        chosen_depot_idx = remaining_nodes_idx[chosen_depot_idx_]
        chosen_nodes_idx = np.concatenate([chosen_depot_idx, chosen_nodes_idx], axis=0)

    coords = coords_L[chosen_nodes_idx]
    distance_matrix = distance_matrix_L[chosen_nodes_idx, :][:, chosen_nodes_idx]
    haversine_matrix = haversine_matrix_L[chosen_nodes_idx, :][:, chosen_nodes_idx]
    return coords, distance_matrix
    
def generate_time_windows(num_requests, planning_time, time_windows_length, service_durations, distance_matrix):
    pickup_nodes_idx = np.arange(num_requests) + 1
    delivery_nodes_idx = pickup_nodes_idx + num_requests
    pickup_delivery_distance = distance_matrix[pickup_nodes_idx, delivery_nodes_idx]
    depot_pickup_distance = distance_matrix[0, pickup_nodes_idx]
    delivery_depot_distance = distance_matrix[delivery_nodes_idx, 0]
    pickup_service_durations = service_durations[pickup_nodes_idx]
    delivery_service_durations = service_durations[delivery_nodes_idx]
    pickup_tw_lb = depot_pickup_distance
    pickup_tw_ub = planning_time - (pickup_delivery_distance + delivery_depot_distance + pickup_service_durations + delivery_service_durations)
    pickup_tw_mid = np.floor(np.random.random(size=(num_requests))*(pickup_tw_ub-pickup_tw_lb)+pickup_tw_lb)
    pickup_early_time_windows =(pickup_tw_mid - (time_windows_length//2))[:, np.newaxis]
    pickup_late_time_windows = (pickup_tw_mid + (time_windows_length//2))[:, np.newaxis]
    pickup_time_windows = np.concatenate([pickup_early_time_windows, pickup_late_time_windows], axis=1)
    pickup_time_windows = np.clip(pickup_time_windows, 0, planning_time)
    
    # two types of time windows
    overlap_delivery_tw_lb = pickup_time_windows[:, 0] + pickup_delivery_distance + pickup_service_durations
    non_overlap_delivery_tw_lb = pickup_time_windows[:, 1] + pickup_delivery_distance + pickup_service_durations
    non_overlap_idx = np.random.choice(num_requests, size=(int(0.1*num_requests)), replace=False)
    is_non_overlap = np.zeros((num_requests), dtype=bool)
    is_non_overlap[non_overlap_idx] = 1
    delivery_tw_lb = overlap_delivery_tw_lb
    delivery_tw_lb[is_non_overlap] = non_overlap_delivery_tw_lb[is_non_overlap]
    delivery_tw_ub = (delivery_tw_lb + time_windows_length)[:, np.newaxis]
    delivery_tw_lb = delivery_tw_lb[:, np.newaxis]
    delivery_time_windows = np.concatenate([delivery_tw_lb, delivery_tw_ub], axis=1)
    # some of the request get no time windows
    z = np.random.random()*0.10 + 0.05
    num_requests_no_tw = int(z*num_requests)
    if num_requests_no_tw > 0:
        requests_no_tw_idx = np.random.choice(num_requests, size=(num_requests_no_tw), replace=False)
        pickup_time_windows[requests_no_tw_idx, 0] = 0
        pickup_time_windows[requests_no_tw_idx, 1] = planning_time
        delivery_time_windows[requests_no_tw_idx,0] = 0
        delivery_time_windows[requests_no_tw_idx,1] = planning_time
    time_windows = np.concatenate([pickup_time_windows, delivery_time_windows], axis=0)
    time_windows = np.insert(time_windows, 0, [0, planning_time], axis=0)
    return time_windows

