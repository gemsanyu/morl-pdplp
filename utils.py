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
    sum_logprobs = torch.zeros((batch_size,), device=agent.device, dtype=torch.float32)
    sum_entropies = torch.zeros((batch_size,), device=agent.device, dtype=torch.float32)
    
    static_features, vehicle_dynamic_features, node_dynamic_features, feasibility_mask = env.begin()
    static_features = torch.from_numpy(static_features).to(agent.device)
    vehicle_dynamic_features = [torch.from_numpy(vehicle_dynamic_features[i]).to(agent.device) for i in range(env.batch_size)] 
    node_dynamic_features = [torch.from_numpy(node_dynamic_features[i]).to(agent.device) for i in range(env.batch_size)]
    feasibility_mask = [torch.from_numpy(feasibility_mask[i]).to(agent.device) for i in range(env.batch_size)]

    active_batch_idx = np.asanyarray([i for i in range(batch_size) if torch.any(feasibility_mask[i])])
    while len(active_batch_idx) > 0:
        active_prev_node_embeddings = [node_embeddings[i,env.current_location_idx[i],:] for i in active_batch_idx]
        active_node_embeddings = node_embeddings[active_batch_idx]
        active_node_dynamic_features = [node_dynamic_features[i] for i in active_batch_idx]
        active_vehicle_dynamic_features = [vehicle_dynamic_features[i] for i in active_batch_idx]
        active_glimpse_V_static = glimpse_V_static[:, active_batch_idx]
        active_glimpse_K_static = glimpse_K_static[:, active_batch_idx]
        active_logits_K_static = logits_K_static[active_batch_idx]
        active_feasibility_mask = [feasibility_mask[i] for i in active_batch_idx]
        active_num_vehicles = env.num_vehicles[active_batch_idx]
        active_fixed_context = fixed_context[active_batch_idx]
        forward_results = agent.forward(active_num_vehicles,
                                        active_node_embeddings,
                                        active_fixed_context,
                                        active_prev_node_embeddings,
                                        active_node_dynamic_features,
                                        active_vehicle_dynamic_features,
                                        active_glimpse_V_static,
                                        active_glimpse_K_static,
                                        active_logits_K_static,
                                        active_feasibility_mask,
                                        param_dict=None)
        selected_vec, selected_node, logprob_list, entropy_list = forward_results
        env.act(active_batch_idx, selected_vec, selected_node)
        print(selected_vec,selected_node)
        exit()