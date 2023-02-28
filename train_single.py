import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from arguments import get_parser
from bpdplp.bpdplp_env import BPDPLP_Env
from bpdplp.bpdplp_dataset import BPDPLP_Dataset
from model.agent import Agent
from utils import encode, solve_decode_only
from setup import setup

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

def run(args):
    agent, opt, tb_writer, last_epoch = setup(args)
    dataset = BPDPLP_Dataset(num_requests=50, num_vehicles_list=[4])
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
    args = prepare_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)