import math
from multiprocessing import Pool
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

from agent.agent import Agent
from arguments import get_parser
from policy.hv import Hypervolume
from utils import write_test_phn_progress, encode, solve_decode_only
from setup_r1nes import setup_r1_nes
from validator import load_validator, save_validator

CPU_DEVICE = torch.device("cpu")

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

@torch.no_grad()
def validate(agent, policy, validation_env_list, pop_size=10):
    # sample once for every env?
    agent.eval()
    num_env = len(validation_env_list)
    batch_size = validation_env_list[0].batch_size
    
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
    solution_list = torch.zeros((num_env, batch_size, pop_size, 2), dtype=torch.float32)    

    for env_idx, env in enumerate(tqdm(validation_env_list)):
        static_features = env.get_static_features()
        num_nodes, num_items, batch_size = env.num_nodes, env.num_items, env.batch_size
        encode_output = encode(agent, static_features, num_nodes, num_items, batch_size)
        static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_output
    
        travel_time_list = torch.zeros((batch_size, pop_size, 1), dtype=torch.float32)
        total_profit_list = torch.zeros((batch_size, pop_size, 1), dtype=torch.float32)
    
        for n, param_dict in enumerate(param_dict_list):
            solve_output = solve_decode_only(agent, env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
            tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve_output
            travel_time_list[:, n, 0] = tour_lengths
            total_profit_list[:, n, 0] = total_profits
        env_solution_list = torch.cat([travel_time_list, total_profit_list], dim=-1)
        solution_list[env_idx] = env_solution_list
    solution_list[:,:,:,1] = -solution_list[:,:,:,1]
    return solution_list    

def compute_hv(solution_list, nadir_points, utopia_points):
    # print(solution_list.shape)
    num_env, batch_size, _, _ = solution_list.shape
    new_nadir_points = torch.zeros_like(nadir_points)
    new_utopia_points = torch.zeros_like(utopia_points)
    new_hv = torch.zeros((num_env, batch_size, 1))
    reference_point = np.array([1.1,1.1])
    hv_getter = Hypervolume(reference_point)
    for env_idx in range(num_env):
        for batch_idx in range(batch_size):
            env_solutions = solution_list[env_idx, batch_idx]
            new_nadir_point, _ = torch.max(env_solutions, dim=0)
            new_utopia_point, _ = torch.min(env_solutions, dim=0)
            old_nadir_point = nadir_points[env_idx, batch_idx]
            old_utopia_point = utopia_points[env_idx, batch_idx]
            new_nadir_point = torch.maximum(new_nadir_point, old_nadir_point)
            new_utopia_point = torch.minimum(new_utopia_point, old_utopia_point)
            new_nadir_points[env_idx, batch_idx] = new_nadir_point
            new_utopia_points[env_idx, batch_idx] = new_utopia_point
            new_nadir_point, new_utopia_point = new_nadir_point.unsqueeze(0), new_utopia_point.unsqueeze(0)
            _N = (env_solutions-new_utopia_point)/(new_nadir_point-new_utopia_point)
            hv = hv_getter.calc(_N.numpy())
            new_hv[env_idx, batch_idx] = hv
    
    return new_hv, new_nadir_points, new_utopia_points
    
@torch.no_grad()
def test_one_epoch(agent, policy, test_env, pop_size=50):
    agent.eval()
    # reuse the static embeddings
    #get static embeddings first, it can be widely reused
    static_features, _, _, _ = test_env.begin()
    num_nodes, num_items, batch_size = test_env.num_nodes, test_env.num_items, test_env.batch_size
    encode_output = encode(agent, static_features, num_nodes, num_items, batch_size)
    static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_output

    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
    solution_list = []
    for param_dict in tqdm(param_dict_list):
        for k in param_dict.keys():
            param_dict[k] = param_dict[k].to(agent.device)
        solve_output = solve_decode_only(agent, test_env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve_output
        solution_list += [torch.stack([tour_lengths, total_profits], dim=1)]
    solution_list = torch.cat(solution_list)
    return solution_list
    
def run(args):
    agent, policy, _, writer, _, test_env, test_sample_solutions  = setup_r1_nes(args)
    vd = load_validator(args.title)
    solution_list = validate(agent, policy, vd.validation_env_list)
    new_hv, new_nadir_points, new_utopia_points = compute_hv(solution_list, vd.nadir_points, vd.utopia_points) 
    vd.update_ref_points(new_nadir_points, new_utopia_points)
    vd.insert_hv_history(new_hv)
    vd.epoch += 1
    test_solution_list = test_one_epoch(agent, policy, test_env)
    write_test_phn_progress(writer, test_solution_list, vd.epoch, args.dataset_name, test_sample_solutions)
    writer.add_scalar("Validation Mean HV", vd.last_hv)
    save_validator(vd, args.title)

if __name__ == '__main__':
    args = prepare_args()
    # torch.set_num_threads(os.cpu_count())
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)