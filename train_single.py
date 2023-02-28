import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from bpdplp.bpdplp_env import BPDPLP_Env
from bpdplp.bpdplp_dataset import BPDPLP_Dataset
from model.agent import Agent
from utils import encode, solve_decode_only

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
    dataset = BPDPLP_Dataset(num_requests=8, num_vehicles_list=[4])
    dl = DataLoader(dataset, batch_size=1)
    for i, batch in enumerate(dl):
        num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types = batch
        env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
        static_features,_,_,_ = env.begin()
        static_features = torch.from_numpy(static_features).to(agent.device)
        encode_results = encode(agent, static_features)
        node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_results
        solve_results = solve_decode_only(agent, env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static)
        tour_list, departure_time_list, logprobs, total_costs, total_penalties = solve_results
        # print(fixed_context, glimpse_K_static)
        exit()        

if __name__ == "__main__":
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    run()