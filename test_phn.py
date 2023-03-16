import pathlib
import random
import time

import numpy as np
import torch

from bpdplp.bpdplp_env import BPDPLP_Env
from utils_moo import prepare_args, generate_params
from utils import encode, solve_decode_only
from setup_phn import setup_phn

"""
actually copying solve_one_batch, but is done so that 
encoding and decoding can be timed separately
"""
@torch.no_grad()
def test(agent, phn, test_batch, x_file, y_file, t_file, num_ray=200):
    agent.eval()
    total_start = time.time()
    ray_list, param_dict_list = generate_params(phn, num_ray, agent.device)
    num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types = test_batch
    env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
    static_features,_,_,_ = env.begin()
    encode_start = time.time()
    static_features = torch.from_numpy(static_features).to(agent.device)
    encode_results = encode(agent, static_features)
    node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_results
    encode_duration = time.time()-encode_start

    decode_start = time.time()
    batch_f_list = [] 
    logprob_list = []
    for param_dict in param_dict_list:
        solve_results = solve_decode_only(agent, env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
        tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties, sum_logprobs, sum_entropies = solve_results
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
    agent, phn, _, validator, tb_writer, test_batch, _, _ = setup_phn(args, validation=True)
    result_dir = pathlib.Path(".")/"result"/args.title
    result_dir.mkdir(parents=True, exist_ok=True)
    title = args.test_instance_name +"-"+str(args.test_num_vehicles) + "-" + args.title
    y_file_path = result_dir/(title+".y")
    x_file_path = result_dir/(title+".x")
    t_file_path = result_dir/(title+".t")
    with open(x_file_path.absolute(), "w") as x_file, open(y_file_path.absolute(), "w") as y_file, open(t_file_path.absolute(), "w") as t_file:
        test(agent,phn,test_batch,x_file,y_file,t_file,args.num_ray)

if __name__ == "__main__":
    args = prepare_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)