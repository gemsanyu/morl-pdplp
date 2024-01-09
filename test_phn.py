import pathlib
import random
import time

import numba
import numpy as np
import torch
import torch.nn.functional as F

from bpdplp.bpdplp_env import BPDPLP_Env
from model.agent import Agent
from utils_moo import get_ray_list
from utils import encode_strict, solve_decode_only, prepare_args
from setup import setup

"""
actually copying solve_one_batch, but is done so that 
encoding and decoding can be timed separately
"""
@torch.no_grad()
def test(agent:Agent, test_batch, x_file, y_file, t_file, num_ray=200):
    agent.eval()
    total_start = time.time()
    ray_list =  get_ray_list(num_ray, agent.device, is_random=False)
    idx, num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types = test_batch
    env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
    static_features,_,_,_ = env.begin()
    encode_start = time.time()
    static_features = torch.from_numpy(static_features).to(agent.device)
    encode_results = encode_strict(agent, static_features)
    node_embeddings, graph_embeddings = encode_results
    encode_duration = time.time()-encode_start

    decode_start = time.time()
    batch_f_list = [] 
    logprob_list = []
    for i, ray in enumerate(ray_list):
        agent.get_param_dict(ray)
        fixed_context = F.linear(graph_embeddings, agent.pf_weight)
        projected_embeddings = F.linear(node_embeddings, agent.pe_weight)
        glimpse_K_static, glimpse_V_static, logits_K_static = projected_embeddings.chunk(3, dim=-1)
        glimpse_K_static = agent._make_heads(glimpse_K_static)
        glimpse_V_static = agent._make_heads(glimpse_V_static)
        solve_results = solve_decode_only(agent, env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static)
        tour_list, _, _, travel_costs, late_penalties, _, sum_logprobs, _ = solve_results
        x_file.write(str(i)+" "+str(len(tour_list[0]))+"\n")
        for tour in tour_list[0]:
            tour_str = ""
            for node in tour:
                tour_str += str(node) + " "
            x_file.write(tour_str+"\n")
        f_list = np.concatenate([travel_costs[:,np.newaxis,np.newaxis], late_penalties[:,np.newaxis,np.newaxis]], axis=2)
        batch_f_list += [f_list]
        logprob_list += [sum_logprobs.unsqueeze(1)]
    batch_f_list = np.concatenate(batch_f_list, axis=1)
    logprob_list = torch.cat(logprob_list, dim=1)
    decode_duration = time.time()-decode_start
    total_duration = time.time()-total_start

    # now write to corresponding file
    for f in batch_f_list[0,:]:
        travel_cost_str ="{:.16f}".format(f[0])
        late_penalty_str="{:.16f}".format(f[1])
        y_file.write((travel_cost_str +" "+late_penalty_str+"\n"))

    # print x? apanya yang diprint....
    total_duration = "{:.16f}".format(total_duration)
    encode_duration = "{:.16f}".format(encode_duration)
    decode_duration = "{:.16f}".format(decode_duration)
    t_file.write((total_duration+" "+encode_duration+" "+decode_duration+"\n"))

    

def run(args):
    agent_, agent, _, _, _, _, _, test_batch, _ = setup(args)
    result_dir = pathlib.Path(".")/"result"/args.title
    result_dir.mkdir(parents=True, exist_ok=True)
    title = args.test_instance_name +"-"+str(args.test_num_vehicles) + "-" + args.title
    y_file_path = result_dir/(title+".y")
    y_file_path.parents[0].mkdir(parents=True, exist_ok=True)
    x_file_path = result_dir/(title+".x")
    t_file_path = result_dir/(title+".t")
    with open(x_file_path.absolute(), "w") as x_file, open(y_file_path.absolute(), "w") as y_file, open(t_file_path.absolute(), "w") as t_file:
        test(agent,test_batch,x_file,y_file,t_file,args.num_ray)

if __name__ == "__main__":
    args = prepare_args()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    numba.set_num_threads(8)
    run(args)