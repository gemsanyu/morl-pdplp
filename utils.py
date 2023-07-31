import os
import pathlib
import sys

import numpy as np
import torch

from arguments import get_parser
from bpdplp.bpdplp import BPDPLP
from bpdplp.bpdplp_env import BPDPLP_Env

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

def instance_to_batch(instance:BPDPLP)->BPDPLP_Env:
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
    max_capacity = torch.tensor([instance.max_capacity])
    num_vehicles = torch.tensor([instance.num_vehicles])
    planning_time = torch.tensor([instance.planning_time])
    # env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time , time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
    # return env
    return num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types
    
# joint TRL Encoder
# encode spatial+other first
# then encode time windows + spatial
def encode(agent, static_features):
    num_requests = int((static_features.shape[1]-1)//2)
    depot_static_features = static_features[:, 0].unsqueeze(1)
    delivery_static_features = static_features[:,num_requests+1:]
    pickup_only_static_features = static_features[:,1:num_requests+1]
    
    depot_spatial_other_features = depot_static_features[:,:,:4]
    delivery_spatial_other_features = delivery_static_features[:,:,:4]
    pickup_spatial_other_features = torch.cat([pickup_only_static_features[:,:,:4], delivery_static_features[:,:,:4]], dim=2) 
    depot_so_init_embedding = agent.depot_spatial_other_embedder(depot_spatial_other_features)
    pickup_so_init_embedding = agent.pick_spatial_other_embedder(pickup_spatial_other_features)
    delivery_so_init_embedding = agent.delivery_spatial_other_embedder(delivery_spatial_other_features)
    node_so_init_embeddings = torch.cat([depot_so_init_embedding, pickup_so_init_embedding, delivery_so_init_embedding], dim=1)
    node_so_embeddings, graph_so_embeddings = agent.spatial_other_gae(node_so_init_embeddings)
    
    depot_time_windows = depot_static_features[:,:,4:]
    delivery_time_windows = delivery_static_features[:,:,4:]
    pickup_time_windows = torch.cat([pickup_only_static_features[:,:,4:], delivery_time_windows], dim=2)
    depot_temporal_init_embedding = agent.depot_temporal_embedder(depot_time_windows)
    pickup_temporal_init_embedding = agent.pick_temporal_embedder(pickup_time_windows)
    delivery_temporal_init_embedding = agent.delivery_temporal_embedder(delivery_time_windows)
    node_temporal_init_embeddings = torch.cat([depot_temporal_init_embedding, pickup_temporal_init_embedding, delivery_temporal_init_embedding], dim=1)
    node_init_embeddings = torch.cat([node_so_embeddings, node_temporal_init_embeddings], dim=2)
    node_temporal_embeddings, graph_temporal_embeddings = agent.temporal_gae(node_init_embeddings)
    
    node_embeddings = node_so_embeddings + node_temporal_embeddings
    graph_embeddings = graph_so_embeddings + graph_temporal_embeddings
    fixed_context = agent.project_fixed_context(graph_embeddings)
    glimpse_K_static, glimpse_V_static, logits_K_static = agent.project_embeddings(node_embeddings).chunk(3, dim=-1)
    glimpse_K_static = agent._make_heads(glimpse_K_static)
    glimpse_V_static = agent._make_heads(glimpse_V_static)
    return node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static



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
def solve_decode_only(agent, env:BPDPLP_Env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict=None):
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
                                        feasibility_mask,
                                        param_dict)
        selected_vecs, selected_nodes, logprobs, entropy_list = forward_results
        selected_vecs = selected_vecs.cpu().numpy()
        selected_nodes = selected_nodes.cpu().numpy()
        vehicle_dynamic_features, node_dynamic_features, feasibility_mask, reward = env.act(batch_idx, selected_vecs, selected_nodes)
        logprob_list += [logprobs[:, np.newaxis]]
        reward_list += [reward[:, np.newaxis, :]]
        vehicle_dynamic_features = torch.from_numpy(vehicle_dynamic_features).to(agent.device, dtype=torch.float32)
        node_dynamic_features = torch.from_numpy(node_dynamic_features).to(agent.device, dtype=torch.float32)
        feasibility_mask = torch.from_numpy(feasibility_mask).to(agent.device, dtype=bool)
        
    reward_list = np.concatenate(reward_list, axis=1)
    logprob_list = torch.concatenate(logprob_list, dim=1)
    tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties = env.finish()
    return tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties, reward_list, logprob_list, sum_entropies

def update(agent, opt, loss, max_grad_norm):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)

def update_step_only(agent, opt, max_grad_norm):
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=max_grad_norm)
    opt.step()
    opt.zero_grad(set_to_none=True)
        

def save(agent, opt, best_validation_score, best_agent_state_dict, epoch, title):
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(title+".pt")
    checkpoint = {
        "agent_state_dict":agent.state_dict(),
        "agent_opt_state_dict":opt.state_dict(),  
        "best_validation_score":best_validation_score,
        "best_agent_state_dict": best_agent_state_dict,
        "epoch":epoch,
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())
