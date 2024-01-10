import pathlib
import random
import time

import numpy as np
import numba
import torch

from bpdplp.bpdplp_env import BPDPLP_Env
from bpdplp.bpdplp_dataset import BPDPLP_Dataset
from utils_moo import generate_params, solve_one_batch, get_ray_list
from utils import encode, solve_decode_only, prepare_args
from setup_phn import setup_phn

"""
actually copying solve_one_batch, but is done so that 
encoding and decoding can be timed separately
"""
@torch.no_grad()
def test(agent, phn, test_batch, x_file, y_file, t_file, num_ray=200):
    agent.eval()
    total_start = time.time()
    ray_list =  get_ray_list(num_ray, agent.device)
    param_dict_list = generate_params(phn, ray_list)
    _, num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types = test_batch
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
    things_to_write = ""
    for i, param_dict in enumerate(param_dict_list):
        print(i)
        solve_results = solve_decode_only(agent, env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
        tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties, reward_list, sum_logprobs, sum_entropies = solve_results
        f_list = np.concatenate([travel_costs[:,np.newaxis,np.newaxis], late_penalties[:,np.newaxis,np.newaxis]], axis=2)
        batch_f_list += [f_list]
        header = str(i)+" "+str(len(tour_list[0]))+"\n"
        things_to_write += header
        # x_file.write()
        for tour in tour_list[0]:
            tour_str = ""
            for node in tour:
                tour_str += str(node) + " "
            tour_str += "\n"
            things_to_write += tour_str
            # x_file.write(tour_str+"\n")
    x_file.write(things_to_write)
    print("written x")
        # for departure_time in departure_time_list[0]:
        #     departure_time_str = ""
        #     for dt in departure_time:
        #         departure_time_str += ("{:.3f}".format(dt)+" ")
        #     x_file.write((departure_time_str + "\n"))


    logprob_list += [sum_logprobs.unsqueeze(1)]
    batch_f_list = np.concatenate(batch_f_list, axis=1)
    logprob_list = torch.cat(logprob_list, dim=1)
    decode_duration = time.time()-decode_start
    total_duration = time.time()-total_start

    

    # now write to corresponding file
    things_to_write = ""
    for f in batch_f_list[0,:]:
        travel_cost_str ="{:.16f}".format(f[0])
        late_penalty_str="{:.16f}".format(f[1])
        y_line = travel_cost_str +" "+late_penalty_str+"\n"
        things_to_write += y_line
    y_file.write(things_to_write)

    # print x? apanya yang diprint....
    total_duration = "{:.16f}".format(total_duration)
    encode_duration = "{:.16f}".format(encode_duration)
    decode_duration = "{:.16f}".format(decode_duration)
    t_file.write((total_duration+" "+encode_duration+" "+decode_duration+"\n"))


def run(args):
    agent, _, phn, _, _, _, _, _, test_batch, _ = setup_phn(args, validation=True)
    result_dir = pathlib.Path(".")/"result"/args.title
    result_dir.mkdir(parents=True, exist_ok=True)
    title = args.test_instance_name +"-"+str(args.test_num_vehicles) + "-" + args.title
    y_file_path = result_dir/(title+".y")
    y_file_path.parents[0].mkdir(parents=True, exist_ok=True)
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
    numba.set_num_threads(1)
    run(args)
