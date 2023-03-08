import os
import pathlib

import numpy as np
import torch

from bpdplp.bpdplp_env import BPDPLP_Env
from model.agent import Agent
# joint TRL Encoder
# encode spatial+other first
# then encode time windows + spatial
def encode(agent:Agent, static_features):
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
                                        param_dict=None)
        selected_vecs, selected_nodes, logprob_list, entropy_list = forward_results
        selected_vecs = selected_vecs.cpu().numpy()
        selected_nodes = selected_nodes.cpu().numpy()
        env.act(batch_idx, selected_vecs, selected_nodes)
        sum_logprobs += logprob_list
        sum_entropies += entropy_list
        vehicle_dynamic_features, node_dynamic_features, feasibility_mask = env.get_state()
        vehicle_dynamic_features = torch.from_numpy(vehicle_dynamic_features).to(agent.device, dtype=torch.float32)
        node_dynamic_features = torch.from_numpy(node_dynamic_features).to(agent.device, dtype=torch.float32)
        feasibility_mask = torch.from_numpy(feasibility_mask).to(agent.device, dtype=bool)
        
    tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties = env.finish()
    return tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties, sum_logprobs, sum_entropies

def save(agent, agent_opt, validation_score, epoch, title):
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(title+".pt")
    checkpoint = {
        "agent_state_dict":agent.state_dict(),
        "agent_opt_state_dict":agent_opt.state_dict(),  
        "validation_score":validation_score,
        "epoch":epoch,
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())

    # saving best checkpoint
    best_checkpoint_path = checkpoint_path.parent /(checkpoint_path.name + "_best")
    if not os.path.isfile(best_checkpoint_path.absolute()):
        torch.save(checkpoint, best_checkpoint_path)
    else:
        best_checkpoint =  torch.load(best_checkpoint_path.absolute())
        best_validation_score = best_checkpoint["validation_score"]
        if best_validation_score < validation_score:
            torch.save(checkpoint, best_checkpoint_path.absolute())
   
