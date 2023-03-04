import os
import pathlib

import numpy as np
import torch


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
    fixed_context = agent.project_fixed_context(graph_embeddings)
    glimpse_K_static, glimpse_V_static, logits_K_static = agent.project_embeddings(node_embeddings).chunk(3, dim=-1)
    glimpse_K_static = agent._make_heads(glimpse_K_static)
    glimpse_V_static = agent._make_heads(glimpse_V_static)
    return node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static

def solve_decode_only(agent, env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static):
    batch_size, num_nodes = env.batch_size, env.num_nodes
    batch_idx = np.arange(batch_size)
    sum_logprobs = torch.zeros((batch_size,), device=agent.device, dtype=torch.float32)
    sum_entropies = torch.zeros((batch_size,), device=agent.device, dtype=torch.float32)
    static_features, vehicle_dynamic_features, node_dynamic_features, feasibility_mask = env.begin()
    vehicle_dynamic_features = torch.from_numpy(vehicle_dynamic_features).to(agent.device)
    node_dynamic_features = torch.from_numpy(node_dynamic_features).to(agent.device)
    feasibility_mask = torch.from_numpy(feasibility_mask).to(agent.device)
    num_vehicles = torch.from_numpy(env.num_vehicles).to(dtype=torch.long)
    num_vehicles_cum = torch.cat([torch.tensor([0]),torch.cumsum(num_vehicles, dim=0)])
    total_num_vehicles = int(num_vehicles_cum[-1])
    vehicle_batch_idx = np.concatenate([ np.asanyarray([i]*num_vehicles[i]) for i in range(batch_size)])
    # reshape these accordingly
    # prepare the static to be repeated as many as the number of vehicles
    # in each batch size
    # repeat fixed context for each vehicle    
    glimpse_V_static = glimpse_V_static[:,vehicle_batch_idx]
    glimpse_K_static = glimpse_K_static[:,vehicle_batch_idx]
    logits_K_static = logits_K_static[vehicle_batch_idx]
    fixed_context = fixed_context[vehicle_batch_idx].unsqueeze(1)
    while torch.any(feasibility_mask):
        current_location_idx = np.concatenate([env.current_location_idx[i] for i in range(batch_size)])
        prev_node_embeddings = node_embeddings[vehicle_batch_idx,current_location_idx,:]
        forward_results = agent.forward(num_vehicles_cum,
                                        total_num_vehicles,
                                        node_embeddings,
                                        fixed_context,
                                        prev_node_embeddings,
                                        node_dynamic_features,
                                        vehicle_dynamic_features,
                                        glimpse_V_static,
                                        glimpse_K_static,
                                        logits_K_static,
                                        feasibility_mask,
                                        param_dict=None)
        selected_vec, selected_node, logprob_list, entropy_list = forward_results
        env.act(batch_idx, selected_vec, selected_node)
        sum_logprobs += logprob_list
        sum_entropies += entropy_list
        vehicle_dynamic_features, node_dynamic_features, feasibility_mask = env.get_state()
        vehicle_dynamic_features = torch.from_numpy(vehicle_dynamic_features).to(agent.device)
        node_dynamic_features = torch.from_numpy(node_dynamic_features).to(agent.device) 
        feasibility_mask = torch.from_numpy(feasibility_mask).to(agent.device)  
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
   
