import os
import pathlib
import sys

import numpy as np
import torch
from torch.nn import functional as F

from arguments import get_parser
from pdptw.pdptw_env import PDPTW_Env
from pdptw.pdptw import PDPTW
from model.agent_so import make_heads

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

def instance_to_batch(instance:PDPTW)->PDPTW_Env:
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
    max_capacity = torch.tensor([instance.max_capacity])
    num_vehicles = torch.tensor([instance.num_vehicles])
    planning_time = torch.tensor([instance.planning_time])
    # env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time , time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
    # return env
    return 0, num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix
    

def encode(agent, static_features, param_dict=None):
    num_requests = int((static_features.shape[1]-1)//2)
    depot_static_features = static_features[:, 0].unsqueeze(1)
    delivery_static_features = static_features[:,num_requests+1:]
    pickup_static_features = torch.concat([static_features[:,1:num_requests+1], delivery_static_features], dim=2)
    depot_init_embedding = agent.depot_embedder(depot_static_features)
    pickup_init_embedding = agent.pick_embedder(pickup_static_features)
    delivery_init_embedding = agent.delivery_embedder(delivery_static_features)
    node_init_embeddings = torch.concat([depot_init_embedding, pickup_init_embedding, delivery_init_embedding], dim=1)
    node_embeddings, graph_embeddings = agent.gae(node_init_embeddings)
    if param_dict is not None:
        fixed_context = F.linear(graph_embeddings, param_dict["pf_weight"])
    else:
        fixed_context = agent.project_fixed_context(graph_embeddings)
    if param_dict is not None:
        projected_embeddings = F.linear(node_embeddings, param_dict["pe_weight"])
    else:
        projected_embeddings = agent.project_embeddings(node_embeddings)
    glimpse_K_static, glimpse_V_static, logits_K_static = projected_embeddings.chunk(3, dim=-1)
    glimpse_K_static = make_heads(glimpse_K_static, agent.n_heads, agent.key_size)
    glimpse_V_static = make_heads(glimpse_V_static, agent.n_heads, agent.key_size)
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

def solve_decode_only(agent, env:PDPTW_Env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict=None):
    batch_size, num_nodes, embed_dim = node_embeddings.shape
    batch_idx = np.arange(batch_size)
    sum_entropies = torch.zeros((batch_size,), device=agent.device, dtype=torch.float32)
    static_features, vehicle_dynamic_features, node_dynamic_features, feasibility_mask = env.begin()
    vehicle_dynamic_features = torch.from_numpy(vehicle_dynamic_features).to(agent.device, dtype=torch.float32)
    node_dynamic_features = torch.from_numpy(node_dynamic_features).to(agent.device, dtype=torch.float32)
    feasibility_mask = torch.from_numpy(feasibility_mask).to(agent.device, dtype=bool)
    num_vehicles = env.num_vehicles
    max_num_vehicles = int(np.max(num_vehicles))
    # num_vehicles_cum = np.concatenate([np.asanyarray([0]),np.cumsum(num_vehicles)])
    # vehicle_batch_idx = np.concatenate([ np.asanyarray([i]*num_vehicles[i]) for i in range(batch_size)])
    # vehicle_idx = np.concatenate([np.arange(num_vehicles[i]) for i in range(batch_size)])
    # expanding glimpses
    glimpse_V_static = glimpse_V_static.unsqueeze(2).expand(-1,-1,max_num_vehicles,-1,-1)
    glimpse_K_static = glimpse_K_static.unsqueeze(2).expand(-1,-1,max_num_vehicles,-1,-1)
    logits_K_static = logits_K_static.unsqueeze(1).expand(-1,max_num_vehicles,-1,-1)
    fixed_context = fixed_context.unsqueeze(1).expand(-1,max_num_vehicles,-1)
    logprob_list = torch.zeros((batch_size, 2*env.num_requests+2), device=agent.device)
    # this time we have to use active batch_idx
    is_still_feasible = np.any(feasibility_mask.cpu().numpy().reshape(batch_size, -1), axis=-1)
    active_batch_idx = batch_idx[is_still_feasible]
    it = 0
    while np.any(is_still_feasible):
        # print("vec features", torch.any(torch.isnan(vehicle_dynamic_features)))
        # print("node features", torch.any(torch.isnan(node_dynamic_features)))
        active_batch_vec_idx = env.batch_vec_idx_not_flat[active_batch_idx]
        active_current_location_idx = env.current_location_idx[active_batch_idx]
        prev_node_embeddings = node_embeddings[active_batch_vec_idx.ravel(), active_current_location_idx.ravel(), :]
        prev_node_embeddings = prev_node_embeddings.view((len(active_batch_idx),max_num_vehicles,-1))
        active_node_embeddings = node_embeddings[active_batch_idx]
        active_fixed_context = fixed_context[active_batch_idx]
        active_node_dynamic_features = node_dynamic_features[active_batch_idx]
        active_vehicle_dynamic_features = vehicle_dynamic_features[active_batch_idx]
        active_glimpse_V_static = glimpse_V_static[:,active_batch_idx,:,:]
        active_glimpse_K_static = glimpse_K_static[:,active_batch_idx,:,:]
        active_logits_K_static = logits_K_static[active_batch_idx]
        active_feasibility_mask = feasibility_mask[active_batch_idx]
        forward_results = agent.forward(active_node_embeddings,
                                        active_fixed_context,
                                        prev_node_embeddings,
                                        active_node_dynamic_features,
                                        active_vehicle_dynamic_features,
                                        active_glimpse_V_static,
                                        active_glimpse_K_static,
                                        active_logits_K_static,
                                        active_feasibility_mask,
                                        param_dict=param_dict)
        selected_vecs, selected_nodes, logprobs, entropy_list = forward_results
        selected_vecs = selected_vecs.cpu().numpy()
        selected_nodes = selected_nodes.cpu().numpy()
        vehicle_dynamic_features, node_dynamic_features, feasibility_mask = env.act(active_batch_idx, selected_vecs, selected_nodes)
        # sum_logprobs += logprobs
        logprob_list[active_batch_idx, it] = logprobs
        sum_entropies[active_batch_idx] += entropy_list
        # vehicle_dynamic_features, node_dynamic_features, feasibility_mask = env.get_state()
        vehicle_dynamic_features = torch.from_numpy(vehicle_dynamic_features).to(agent.device, dtype=torch.float32)
        node_dynamic_features = torch.from_numpy(node_dynamic_features).to(agent.device, dtype=torch.float32)
        is_still_feasible = np.any(feasibility_mask.reshape(batch_size, -1), axis=-1)
        active_batch_idx = batch_idx[is_still_feasible]
        feasibility_mask = torch.from_numpy(feasibility_mask).to(agent.device, dtype=bool)
        it += 1
    tour_list, arrived_time_list, departure_time_list, travel_time, num_node_not_visited = env.finish()
    return tour_list, arrived_time_list, departure_time_list, travel_time, num_node_not_visited, logprob_list, sum_entropies

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
