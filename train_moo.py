import random
import subprocess
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from arguments import get_parser
from bpdplp.bpdplp_env import BPDPLP_Env
from bpdplp.bpdplp_dataset import BPDPLP_Dataset
from model.agent import Agent
from policy.policy import Policy
from policy.utils import get_score_nd_cd, get_score_hv_contributions
from policy.non_dominated_sorting import fast_non_dominated_sort
from utils import encode, solve_decode_only, save
from utils_moo import update_policy, save_policy
from setup_moo import setup_r1nes
from validator import Validator



def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

@torch.no_grad()
def solve_one_batch(args, agent:Agent, param_dict_list, batch):
    num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types = batch
    env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
    static_features,_,_,_ = env.begin()
    static_features = torch.from_numpy(static_features).to(agent.device)
    encode_results = encode(agent, static_features)
    node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_results

    batch_f_list = [] 
    for param_dict in param_dict_list:
        solve_results = solve_decode_only(agent, env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
        tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties, sum_logprobs, sum_entropies = solve_results
        f_list = np.concatenate([travel_costs[:,np.newaxis,np.newaxis], late_penalties[:,np.newaxis,np.newaxis]], axis=2)
        batch_f_list += [f_list]
    batch_f_list = np.concatenate(batch_f_list, axis=1)
    return batch_f_list

@torch.no_grad()        
def validate_one_epoch(args, agent:Agent, policy:Policy, validator:Validator, validation_dataset, test_batch, test_batch2, tb_writer, epoch):
    agent.eval()
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)
    
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=args.pop_size, use_antithetic=False)
    f_list = []
    for batch_idx, batch in tqdm(enumerate(validation_dataloader), desc=f'Validation epoch {epoch}'):
        batch_f_list = solve_one_batch(args, agent, param_dict_list, batch)
        f_list += [batch_f_list] 
    f_list = np.concatenate(f_list,axis=0)
    nadir_points = np.max(f_list, axis=1)
    utopia_points = np.min(f_list, axis=1)
    validator.insert_new_ref_points(nadir_points, utopia_points)

    nd_solutions_list = []
    for i in range(len(validation_dataset)):
        nondom_idx = fast_non_dominated_sort(f_list[i,:,:])[0]
        nd_solutions = f_list[i, nondom_idx, :]
        nd_solutions_list += [nd_solutions]
    validator.insert_new_nd_solutions(nd_solutions_list)
    validator.epoch +=1

    last_mean_running_igd = validator.get_last_mean_running_igd()
    if last_mean_running_igd is not None:
        tb_writer.add_scalar("Mean Running IGD", last_mean_running_igd, validator.epoch)
    last_mean_delta_nadir, last_mean_delta_utopia = validator.get_last_delta_refpoints()
    if last_mean_delta_nadir is not None:
        tb_writer.add_scalar("Mean Delta Nadir", last_mean_delta_nadir, validator.epoch)
        tb_writer.add_scalar("Mean Delta Utopia", last_mean_delta_utopia, validator.epoch)

    # test
    marker_list = [".","o","v","^","<",">","1","2","3","4"]
    colors_list = [key for key in mcolors.TABLEAU_COLORS.keys()]
    combination_list = [[c,m] for c in colors_list for m in marker_list]
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=50, use_antithetic=False)
    test_f_list = solve_one_batch(args, agent, param_dict_list, test_batch)
    plt.figure()
    for i in range(len(param_dict_list)):
        c = combination_list[i][0]
        m = combination_list[i][1]
        plt.scatter(test_f_list[0,i,0], test_f_list[0,i,1], c=c, marker=m)
    tb_writer.add_figure("Solutions "+args.test_instance_name+"-"+str(args.test_num_vehicles), plt.gcf(), validator.epoch)
    
    test_f_list = solve_one_batch(args, agent, param_dict_list, test_batch2)
    plt.figure()
    for i in range(len(param_dict_list)):
        c = combination_list[i][0]
        m = combination_list[i][1]
        plt.scatter(test_f_list[0,i,0], test_f_list[0,i,1], c=c, marker=m)
    tb_writer.add_figure("Solutions bar-n400-1-"+str(args.test_num_vehicles), plt.gcf(), validator.epoch)
    
    return validator
    

@torch.no_grad()        
def train_one_epoch(args, agent:Agent, policy:Policy, validator:Validator, train_dataset, validation_dataset, test_batch, test_batch2, tb_writer, epoch):
    agent.eval()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    vd_proc = None
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc=f'Training epoch {epoch}'):
        print(batch_idx)
        param_dict_list, sample_list = policy.generate_random_parameters(n_sample=args.pop_size, use_antithetic=False)
        batch_f_list = solve_one_batch(args, agent, param_dict_list, batch)
        f_list = [batch_f_list]    
        f_list = np.concatenate(f_list,axis=0)

        score_list = []
        for i in range(args.batch_size):
            score = get_score_hv_contributions(f_list[i,:,:], args.negative_hv)
            score_list += [score[np.newaxis,:,:]]
        score_list = np.concatenate(score_list, axis=0)
        score_list = np.mean(score_list, axis=0)

        policy = update_policy(args.policy, policy, sample_list, score_list)
        policy.write_progress_to_tb(tb_writer)
        if batch_idx % 4 == 0:
            if vd_proc is not None:
                vd_proc.wait()
            save_policy(policy, epoch, args.title)
            # validator = validate_one_epoch(args, agent, policy, validator, validation_dataset, test_batch, test_batch2, tb_writer, epoch)
            # save_validator(validator, args.title)
            vd_proc = validate_no_wait(args)
    save_policy(policy, epoch, args.title)
    vd_proc.wait()

    return policy

def validate_no_wait(args)->subprocess.Popen:
    vd_proc_cmd = ["python",
                    "validate_r1nes.py",
                    "--title",
                    args.title,
                    "--test-instance-name",
                    args.test_instance_name,
                    "--test-num-vehicles",
                    str(args.test_num_vehicles),
                    "--device",
                    "cpu",
                    "--num-validation-samples",
                    str(args.num_validation_samples),
                    "--policy",
                    args.policy]
    vd_proc = subprocess.Popen(vd_proc_cmd)
    return vd_proc

def run(args):
    agent, policy, validator, tb_writer, test_batch, test_batch2, last_epoch = setup_r1nes(args)
    validation_dataset = BPDPLP_Dataset(num_samples=args.num_validation_samples, mode="validation")
    train_dataset = BPDPLP_Dataset(num_samples=args.num_training_samples, mode="training")
    # vd_proc = None
    for epoch in range(last_epoch+1, args.max_epoch):
        policy = train_one_epoch(args, agent, policy, validator, train_dataset, validation_dataset, test_batch, test_batch2, tb_writer, epoch)
        # if vd_proc is not None:
        #     vd_proc.wait()
        # vd_proc = validate_no_wait(args)
        # validator = validate_one_epoch(args, agent, policy, validator, validation_dataset, test_batch, test_batch2, tb_writer, epoch)
        # save_policy(policy, epoch, args.title)
        # save_validator(validator, args.title)
    # vd_proc.wait()

        
if __name__ == "__main__":
    args = prepare_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)