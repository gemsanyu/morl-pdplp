import random
import subprocess
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from tqdm import tqdm


from arguments import get_parser
from bpdplp.bpdplp_env import BPDPLP_Env
from bpdplp.bpdplp_dataset import BPDPLP_Dataset
from model.agent import Agent
from model.phn import PHN
from policy.non_dominated_sorting import fast_non_dominated_sort
from utils import encode, solve_decode_only, save
from utils_moo import save_phn
from setup_phn import setup_phn
from solver.hv_maximization import HvMaximization
from validator import Validator



def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

def solve_one_batch(args, agent:Agent, param_dict_list, batch):
    num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types = batch
    env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
    static_features,_,_,_ = env.begin()
    static_features = torch.from_numpy(static_features).to(agent.device)
    encode_results = encode(agent, static_features)
    node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_results

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
    return logprob_list, batch_f_list

@torch.no_grad()        
def validate_one_epoch(args, agent:Agent, phn:PHN, validator:Validator, validation_dataset, test_batch, test_batch2, tb_writer, epoch):
    agent.eval()
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)
    
    ray_list, param_dict_list = generate_params(phn, args.num_ray, agent.device)
    f_list = []
    for batch_idx, batch in tqdm(enumerate(validation_dataloader), desc=f'Validation epoch {epoch}'):
        logprob_list, batch_f_list = solve_one_batch(args, agent, param_dict_list, batch)
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
    ray_list, param_dict_list = generate_params(phn, 50, agent.device)
    logprobs_list, test_f_list = solve_one_batch(args, agent, param_dict_list, test_batch)
    plt.figure()
    for i in range(len(param_dict_list)):
        c = combination_list[i][0]
        m = combination_list[i][1]
        plt.scatter(test_f_list[0,i,0], test_f_list[0,i,1], c=c, marker=m)
    tb_writer.add_figure("Solutions "+args.test_instance_name+"-"+str(args.test_num_vehicles), plt.gcf(), validator.epoch)
    
    logprobs_list, test_f_list = solve_one_batch(args, agent, param_dict_list, test_batch2)
    plt.figure()
    for i in range(len(param_dict_list)):
        c = combination_list[i][0]
        m = combination_list[i][1]
        plt.scatter(test_f_list[0,i,0], test_f_list[0,i,1], c=c, marker=m)
    tb_writer.add_figure("Solutions bar-n400-1-"+str(args.test_num_vehicles), plt.gcf(), validator.epoch)
    
    return validator
    
def generate_params(phn:PHN, num_ray, device):
    ray_list = []

    param_dict_list = []
    for i in range(num_ray):
        start, end = 0.1, np.pi/2-0.1
        r = np.random.uniform(start + i*(end-start)/num_ray, start+ (i+1)*(end-start)/num_ray)
        ray = np.array([np.cos(r),np.sin(r)], dtype='float32')
        ray /= ray.sum()
        ray *= np.random.randint(1, 5)*abs(np.random.normal(1, 0.2))
        ray = torch.from_numpy(ray).to(device)
        param_dict = phn(ray)
        param_dict_list += [param_dict]
        ray_list += [ray]
    ray_list = torch.stack(ray_list)
    return ray_list, param_dict_list

def get_hv_d(batch_f_list, mo_opt:HvMaximization):
    hv_d_list = [] 
    batch_size, num_sample, num_obj = batch_f_list.shape
    for i in range(batch_size):
        obj_instance = np.transpose(batch_f_list[i,:,:])
        hv_d = mo_opt.compute_weights(obj_instance).transpose(0,1)
        hv_d_list += [hv_d.unsqueeze(0)]
    hv_d_list = torch.cat(hv_d_list, dim=0)
    return hv_d_list

def train_one_epoch(args, agent:Agent, phn:PHN, opt, train_dataset, tb_writer, epoch):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    mo_opt = HvMaximization(n_mo_sol=args.num_ray, n_mo_obj=2)
    
    vd_proc = None
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc=f'Training epoch {epoch}'):
        agent.train()
        ray_list, param_dict_list = generate_params(phn, args.num_ray, agent.device)
        logprob_list, batch_f_list = solve_one_batch(args, agent, param_dict_list, batch)
        agent.eval()
        with torch.no_grad():
            greedy_logprob_list, greedy_batch_f_list = solve_one_batch(args, agent, param_dict_list, batch)
        A = batch_f_list-greedy_batch_f_list
        
        nadir = np.max(A, axis=1, keepdims=True)
        utopia = np.min(A, axis=1, keepdims=True)
        denom = (nadir-utopia)
        denom[denom==0] = 1e-8
        norm_obj = (A-utopia)/denom
        hv_d_list = get_hv_d(A, mo_opt)
        
        # compute loss now
        hv_d_list = hv_d_list.to(agent.device)
        norm_obj = torch.from_numpy(norm_obj).to(agent.device)
        A = torch.from_numpy(A).to(agent.device)
        logprob_list = logprob_list.unsqueeze(2)
        loss_per_obj = logprob_list*A
        final_loss_per_obj = loss_per_obj*hv_d_list
        final_loss_per_instance = final_loss_per_obj.sum(dim=2)
        final_loss_per_ray = final_loss_per_instance.mean(dim=0)
        final_loss = final_loss_per_ray.sum()
        
        ray_list = ray_list.unsqueeze(0).expand_as(loss_per_obj)
        cos_penalty = cosine_similarity(loss_per_obj, ray_list, dim=2)
        cos_penalty_per_ray = cos_penalty.mean(dim=0)
        total_cos_penalty = cos_penalty_per_ray.sum()
        final_loss -= args.ld*total_cos_penalty
        agent.zero_grad(set_to_none=True)
        phn.zero_grad(set_to_none=True)
        opt.zero_grad(set_to_none=True)
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(phn.parameters(), max_norm=1)
        opt.step()
        
        if batch_idx % 4 == 0:
            tb_writer.add_scalar("COS PENALTY", total_cos_penalty.cpu().item())
            if vd_proc is not None:
                vd_proc.wait()
            save_phn(phn, epoch, args.title)
            # validator = validate_one_epoch(args, agent, policy, validator, validation_dataset, test_batch, test_batch2, tb_writer, epoch)
            # save_validator(validator, args.title)
            vd_proc = validate_no_wait(args)
    save_phn(phn, epoch, args.title)        
    vd_proc.wait()


def validate_no_wait(args)->subprocess.Popen:
    vd_proc_cmd = ["python",
                    "validate_phn.py",
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
                    "--num-ray",
                    str(args.num_ray)
                    ]
    vd_proc = subprocess.Popen(vd_proc_cmd)
    return vd_proc

def initialize(po_weight,phn:PHN,opt,tb_writer):
    ray = np.asanyarray([[0.5,0.5]],dtype=float)
    ray = torch.from_numpy(ray).to(phn.device, dtype=torch.float32)
    param_dict = phn(ray)
    param = (param_dict["po_weight"]).ravel()
    loss = torch.norm(po_weight-param)
    opt.zero_grad(set_to_none=True)
    tb_writer.add_scalar("Initialization loss", loss.cpu().item())
    loss.backward()
    opt.step()
    return loss.cpu().item()
    

def run(args):
    agent, phn, opt, validator, tb_writer, test_batch, test_batch2, last_epoch = setup_phn(args)
    # validation_dataset = BPDPLP_Dataset(num_samples=args.num_validation_samples, mode="validation")
    train_dataset = BPDPLP_Dataset(num_samples=args.num_training_samples, mode="training")
    max_init_step = 1000
    po_weight = None
    for name, param in agent.named_parameters():
        if name == "project_out.weight":
            po_weight = param.data.ravel()
            break
    po_weight = po_weight.detach().clone()
    opt_init = torch.optim.Adam(phn.parameters(), lr=1e-4)
    for i in range(max_init_step):
        loss = initialize(po_weight,phn,opt_init,tb_writer)
        if loss < 1e-4:
            break

    for epoch in range(last_epoch+1, args.max_epoch):
        train_one_epoch(args, agent, phn, opt, train_dataset, tb_writer, epoch)

        
if __name__ == "__main__":
    args = prepare_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)