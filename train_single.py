import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from bpdplp.bpdplp_env import BPDPLP_Env
from bpdplp.bpdplp_dataset import BPDPLP_Dataset
from model.agent import Agent

def setup() -> Agent:
    agent = Agent(num_node_static_features=6,
                  num_vehicle_dynamic_features=4,
                  num_node_dynamic_features=1,
                  n_heads=8,
                  n_gae_layers=3,
                  embed_dim=128,
                  gae_ff_hidden=128,
                  tanh_clip=10)
    return agent

def run():
    agent = setup()
    dataset = BPDPLP_Dataset(num_requests=8, num_vehicles_list=[1,2])
    dl = DataLoader(dataset, batch_size=2)
    for i, batch in enumerate(dl):
        num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types = batch
        env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
        static_features, vehicle_dynamic_features, node_dynamic_features, feasibility_mask = env.begin()
        static_features = torch.from_numpy(static_features).to(agent.device)
        vehicle_dynamic_features = [torch.from_numpy(vehicle_dynamic_features[i]).to(agent.device) for i in range(env.batch_size)] 
        node_dynamic_features = [torch.from_numpy(node_dynamic_features[i]).to(agent.device) for i in range(env.batch_size)]
        feasibility_mask = [torch.from_numpy(feasibility_mask[i]).to(agent.device) for i in range(env.batch_size)]
        depot_static_features = static_features[:, 0:1]
        delivery_static_features = static_features[:,env.num_requests+1:]
        pickup_static_features = torch.concat([static_features[:,1:env.num_requests+1], delivery_static_features], dim=2)
        depot_init_embedding = agent.depot_embedder(depot_static_features)
        pickup_init_embedding = agent.pick_embedder(pickup_static_features)
        delivery_init_embedding = agent.delivery_embedder(delivery_static_features)
        node_init_embeddings = torch.concat([depot_init_embedding, pickup_init_embedding, delivery_init_embedding], dim=1)
        node_embeddings, graph_embeddings = agent.gae(node_init_embeddings)
        fixed_context = agent.project_fixed_context(graph_embeddings)
        glimpse_K_static, glimpse_V_static, logits_K_static = agent.project_embeddings(node_embeddings).chunk(3, dim=-1)
        glimpse_K_static = agent._make_heads(glimpse_K_static)
        glimpse_V_static = agent._make_heads(glimpse_V_static)
        prev_node_embeddings = [node_embeddings[i,env.current_location_idx[i],:] for i in range(env.batch_size)]
        
        action, logprob = agent.forward(env.num_vehicles,
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
        # print(fixed_context, glimpse_K_static)
        exit()        

if __name__ == "__main__":
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    run()