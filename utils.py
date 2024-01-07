import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

from arguments import get_parser
from bpdplp.bpdplp_env import BPDPLP_Env
from bpdplp.bpdplp import BPDPLP

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

def instance_to_batch(instance:BPDPLP):
    coords = torch.from_numpy(instance.coords).unsqueeze(0)
    norm_coords = torch.from_numpy(instance.norm_coords).unsqueeze(0)
    demands = torch.from_numpy(instance.demands).unsqueeze(0)
    norm_demands = torch.from_numpy(instance.norm_demands).unsqueeze(0)
    time_windows = torch.from_numpy(instance.time_windows).unsqueeze(0)
    norm_time_windows = torch.from_numpy(instance.norm_time_windows).unsqueeze(0)
    service_durations = torch.from_numpy(instance.service_durations).unsqueeze(0)
    norm_service_durations = torch.from_numpy(instance.norm_service_durations).unsqueeze(0)
    distance_matrix = torch.from_numpy(instance.distance_matrix).unsqueeze(0)
    norm_distance_matrix = torch.from_numpy(instance.norm_distance_matrix).unsqueeze(0)
    road_types = torch.from_numpy(instance.road_types).unsqueeze(0)
    max_capacity = torch.tensor([float(instance.max_capacity)])
    num_vehicles = torch.tensor([instance.num_vehicles])
    planning_time = torch.tensor([float(instance.planning_time)])
    speed_profiles = torch.tensor(instance.speed_profiles)
    time_horizons = torch.tensor(instance.time_horizons)
    # env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time , time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
    # return env
    return 0, num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types, speed_profiles, time_horizons 
    

def encode(agent, static_features):
    num_requests = int((static_features.shape[1]-1)//2)
    depot_static_features = static_features[:, 0].unsqueeze(1)
    delivery_static_features = static_features[:,num_requests+1:]
    pickup_static_features = torch.concat([static_features[:,1:num_requests+1], delivery_static_features], dim=2)
    depot_init_embedding = agent.depot_embedder(depot_static_features)
    pickup_init_embedding = agent.pick_embedder(pickup_static_features)
    delivery_init_embedding = agent.delivery_embedder(delivery_static_features)
    node_init_embeddings = torch.concat([depot_init_embedding, pickup_init_embedding, delivery_init_embedding], dim=1)
    node_embeddings, graph_embeddings = agent.gae(node_init_embeddings)
    fixed_context = F.linear(graph_embeddings, agent.pf_weight)
    projected_embeddings = F.linear(node_embeddings, agent.pe_weight)
    glimpse_K_static, glimpse_V_static, logits_K_static = projected_embeddings.chunk(3, dim=-1)
    glimpse_K_static = agent._make_heads(glimpse_K_static)
    glimpse_V_static = agent._make_heads(glimpse_V_static)
    return node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static

def encode_strict(agent, static_features):
    num_requests = int((static_features.shape[1]-1)//2)
    depot_static_features = static_features[:, 0].unsqueeze(1)
    delivery_static_features = static_features[:,num_requests+1:]
    pickup_static_features = torch.concat([static_features[:,1:num_requests+1], delivery_static_features], dim=2)
    depot_init_embedding = agent.depot_embedder(depot_static_features)
    pickup_init_embedding = agent.pick_embedder(pickup_static_features)
    delivery_init_embedding = agent.delivery_embedder(delivery_static_features)
    node_init_embeddings = torch.concat([depot_init_embedding, pickup_init_embedding, delivery_init_embedding], dim=1)
    node_embeddings, graph_embeddings = agent.gae(node_init_embeddings)
    return node_embeddings, graph_embeddings


"""
the features/components associated with vehicles
must be padded with zeros/dummy vehicles,
so that they have the same dimensions across batches
so that the operations in the forward is more effective
1. prev node embeddings
2. feasibility masking
3. vehicle dynamic features
4. node dynamic features
among the three, prev node embeddings are worrisome
i hope it doesn't affect the agent training,
i think it will not because we will mask the glimpse computation
too. i guess.
"""
def solve_decode_only(agent, env:BPDPLP_Env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static):
    batch_size, num_nodes, embed_dim = node_embeddings.shape
    batch_idx = np.arange(batch_size)
    sum_logprobs = torch.zeros((batch_size,), device=agent.device, dtype=torch.float32)
    sum_entropies = torch.zeros((batch_size,), device=agent.device, dtype=torch.float32)
    static_features, vehicle_dynamic_features, node_dynamic_features, feasibility_mask = env.begin()
    vehicle_dynamic_features = torch.from_numpy(vehicle_dynamic_features).to(agent.device, dtype=torch.float32)
    node_dynamic_features = torch.from_numpy(node_dynamic_features).to(agent.device, dtype=torch.float32)
    feasibility_mask = torch.from_numpy(feasibility_mask).to(agent.device, dtype=bool)
    num_vehicles = env.num_vehicles
    max_num_vehicles = int(np.max(num_vehicles))
    num_vehicles_cum = np.concatenate([np.asanyarray([0]),np.cumsum(num_vehicles)])
    vehicle_batch_idx = np.concatenate([ np.asanyarray([i]*num_vehicles[i]) for i in range(batch_size)])
    vehicle_idx = np.concatenate([np.arange(num_vehicles[i]) for i in range(batch_size)])
    # expanding glimpses
    glimpse_V_static = glimpse_V_static.unsqueeze(2).expand(-1,-1,max_num_vehicles,-1,-1)
    glimpse_K_static = glimpse_K_static.unsqueeze(2).expand(-1,-1,max_num_vehicles,-1,-1)
    logits_K_static = logits_K_static.unsqueeze(1).expand(-1,max_num_vehicles,-1,-1)
    fixed_context = fixed_context.unsqueeze(1).expand(-1,max_num_vehicles,-1)
    reward_list = []
    logprob_list = []
    while torch.any(feasibility_mask):
        prev_node_embeddings = node_embeddings[env.batch_vec_idx, env.current_location_idx.flatten(), :]
        prev_node_embeddings = prev_node_embeddings.view((batch_size,max_num_vehicles,-1))
        forward_results = agent.forward(node_embeddings,
                                        fixed_context,
                                        prev_node_embeddings,
                                        node_dynamic_features,
                                        vehicle_dynamic_features,
                                        glimpse_V_static,
                                        glimpse_K_static,
                                        logits_K_static,
                                        feasibility_mask)
        selected_vecs, selected_nodes, logprobs, entropy_list = forward_results
        selected_vecs = selected_vecs.cpu().numpy()
        selected_nodes = selected_nodes.cpu().numpy()
        vehicle_dynamic_features, node_dynamic_features, feasibility_mask, reward = env.act(batch_idx, selected_vecs, selected_nodes)
        # sum_logprobs += logprobs
        logprob_list += [logprobs[:, np.newaxis]]
        reward_list += [reward[:, np.newaxis, :]]
        sum_entropies += entropy_list
        # vehicle_dynamic_features, node_dynamic_features, feasibility_mask = env.get_state()
        vehicle_dynamic_features = torch.from_numpy(vehicle_dynamic_features).to(agent.device, dtype=torch.float32)
        node_dynamic_features = torch.from_numpy(node_dynamic_features).to(agent.device, dtype=torch.float32)
        feasibility_mask = torch.from_numpy(feasibility_mask).to(agent.device, dtype=bool)
    reward_list = np.concatenate(reward_list, axis=1)
    logprob_list = torch.concatenate(logprob_list, dim=1)
    tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties = env.finish()
    return tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties, reward_list, logprob_list, sum_entropies
