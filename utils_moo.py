import math
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from policy.policy import Policy
from bpdplp.bpdplp_env import BPDPLP_Env
from policy.non_dominated_sorting import fast_non_dominated_sort
from utils import encode, solve_decode_only
from solver.hv_maximization import HvMaximization

def get_hv_d(batch_f_list):
    hv_d_list = [] 
    batch_size, num_sample, num_obj = batch_f_list.shape
    mo_opt = HvMaximization(n_mo_sol=num_sample, n_mo_obj=num_obj)
    for i in range(batch_size):
        obj_instance = np.transpose(batch_f_list[i,:,:])
        hv_d = mo_opt.compute_weights(obj_instance).transpose(0,1)
        hv_d_list += [hv_d.unsqueeze(0)]
    hv_d_list = torch.cat(hv_d_list, dim=0)
    return hv_d_list

def compute_loss(logprob_list, training_nondom_list, idx_list, batch_f_list, greedy_batch_f_list, ray_list):
    device = logprob_list.device
    nadir = []
    utopia = []
    for i in range(len(batch_f_list)):
        nondom_sols =  training_nondom_list[idx_list[i]]
        nadir += [np.max(nondom_sols, axis=0, keepdims=True)[np.newaxis,:]]
        utopia += [np.min(nondom_sols, axis=0, keepdims=True)[np.newaxis,:]]
    nadir = np.concatenate(nadir, axis=0)
    utopia = np.concatenate(utopia, axis=0)
    A = batch_f_list-greedy_batch_f_list
    A = batch_f_list
    denom = (nadir-utopia)
    denom[denom==0] = 1e-8
    norm_obj = (A-utopia)/denom
    norm_obj = batch_f_list
    hv_d_list = get_hv_d(norm_obj)
    
    # compute loss now
    hv_d_list = hv_d_list.to(device)
    norm_obj = torch.from_numpy(norm_obj).to(device)
    A = torch.from_numpy(A.copy()).to(device)
    # A -= A.mean(dim=1,keepdim=True)
    norm_obj -= norm_obj.mean(dim=1, keepdim=True)
    logprob_list = logprob_list.unsqueeze(2)
    loss_per_obj = logprob_list*norm_obj
    final_loss_per_obj = loss_per_obj*hv_d_list
    final_loss_per_instance = final_loss_per_obj.sum(dim=2)
    final_loss_per_ray = final_loss_per_instance.mean(dim=0)
    final_loss = final_loss_per_ray.sum()
    
    ray_list = ray_list.unsqueeze(0).expand_as(loss_per_obj)
    A = torch.from_numpy(batch_f_list).to(logprob_list.device)
    cos_penalty = cosine_similarity(A, ray_list, dim=2)*logprob_list.squeeze(-1)
    # A,B,_ = ray_list.shape
    # for a in range(A):
    #     for b in range(B):
    #         print(ray_list[a,b,:].numpy(), norm_obj[a,b,:].numpy(), cos_penalty[a,b].detach().numpy())
    # exit()
    cos_penalty_per_ray = cos_penalty.mean(dim=0)
    total_cos_penalty = cos_penalty_per_ray.sum()
    return final_loss, total_cos_penalty

def compute_spread_loss(logprobs, training_nondom_list, idx_list, f_list):
    # param_list = [param_dict["v1"].ravel().unsqueeze(0) for param_dict in param_dict_list]
    # param_list = torch.cat(param_list).unsqueeze(0)
    nadir = []
    utopia = []
    for i in range(len(f_list)):
        nondom_sols =  training_nondom_list[idx_list[i]]
        nadir += [np.max(nondom_sols, axis=0, keepdims=True)[np.newaxis,:]]
        utopia += [np.min(nondom_sols, axis=0, keepdims=True)[np.newaxis,:]]
    nadir = np.concatenate(nadir, axis=0)
    utopia = np.concatenate(utopia, axis=0)
    f_list = torch.from_numpy(f_list)
    denom = (nadir-utopia)
    denom[denom==0] = 1e-8
    f_list = (f_list-utopia)/denom
    distance_matrix = torch.cdist(f_list, f_list)
    _, batched_min_distance_per_ray = distance_matrix.min(dim=2)
    # batched_min_distance_per_ray = torch.transpose(batched_min_distance_per_ray,0,1)
    batched_min_distance_per_ray = batched_min_distance_per_ray.to(logprobs.device)
    spread_loss = (logprobs*batched_min_distance_per_ray).mean()
    return spread_loss

def update_phn(agent, phn, opt, final_loss):
    agent.zero_grad(set_to_none=True)
    phn.zero_grad(set_to_none=True)
    opt.zero_grad(set_to_none=True)
    final_loss.backward()
    torch.nn.utils.clip_grad_norm_(phn.parameters(), max_norm=1)
    opt.step()

def generate_params(phn, num_ray, device):
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

def solve_one_batch(agent, param_dict_list, batch, nondom_list):
    idx_list = batch[0]
    batch = batch[1:]
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
    logprob_list = torch.cat(logprob_list, dim=1)
    batch_f_list = np.concatenate(batch_f_list, axis=1)
    if nondom_list is None:
        return logprob_list, batch_f_list, None
    
    for i in range(env.batch_size):
        idx = idx_list[i]
        if nondom_list[idx] is None:
            I = fast_non_dominated_sort(batch_f_list[i,:])[0]
            nondom = batch_f_list[i, I, :]
            nondom_list[idx] = nondom
        else:
            nondom_old = nondom_list[idx]
            nondom_old = np.concatenate([nondom_old, batch_f_list[i,:]])
            I = fast_non_dominated_sort(nondom_old)[0]
            nondom_list[idx] = nondom_old[I]

    return logprob_list, batch_f_list, nondom_list

@torch.no_grad()        
def validate_one_epoch(args, agent, phn, validator, validation_dataset, test_batch, test_batch2, tb_writer, epoch):
    agent.eval()
    # Define the light and dark blue colors
    light_blue = mcolors.CSS4_COLORS['lightblue']
    dark_blue = mcolors.CSS4_COLORS['darkblue']
    matplotlib.use('Agg')

    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size)
    
    ray_list, param_dict_list = generate_params(phn, 50, agent.device)
    f_list = []
    logprobs = []
    for batch_idx, batch in tqdm(enumerate(validation_dataloader), desc=f'Validation epoch {epoch}'):
        logprob_list, batch_f_list, _ = solve_one_batch(agent, param_dict_list, batch, None)
        f_list += [batch_f_list] 
        logprobs += [logprob_list]
    f_list = np.concatenate(f_list,axis=0)
    logprobs = torch.concatenate(logprobs, dim=0)
    # _, total_cos_penalty = compute_loss(logprob_list, norm_f_list, norm_f_list, ray_list)
    # plot 1 or 2 from validation?
    gradient = np.linspace(0,1,len(param_dict_list))
    colors = np.vstack((mcolors.to_rgba(light_blue), mcolors.to_rgba(dark_blue)))
    my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors, N=len(param_dict_list))
    plt.figure()
    plt.scatter(f_list[0,:,0], f_list[0,:,1], c=gradient, cmap=my_cmap)
    tb_writer.add_figure("Solutions Validation 1", plt.gcf(), validator.epoch)
    
    plt.figure()
    plt.scatter(f_list[1,:,0], f_list[1,:,1], c=gradient, cmap=my_cmap)
    tb_writer.add_figure("Solutions Validation 2", plt.gcf(), validator.epoch)
    # validator.epoch+=1
    # nadir_points = np.max(f_list, axis=1)
    # utopia_points = np.min(f_list, axis=1)
    # validator.insert_new_ref_points(nadir_points, utopia_points)

    # nd_solutions_list = []
    # for i in range(len(validation_dataset)):
    #     nondom_idx = fast_non_dominated_sort(f_list[i,:,:])[0]
    #     nd_solutions = f_list[i, nondom_idx, :]
    #     nd_solutions_list += [nd_solutions]
    # validator.insert_new_nd_solutions(nd_solutions_list)
    

    # last_mean_running_igd = validator.get_last_mean_running_igd()
    # if last_mean_running_igd is not None:
    #     tb_writer.add_scalar("Mean Running IGD", last_mean_running_igd, validator.epoch)
    # last_mean_delta_nadir, last_mean_delta_utopia = validator.get_last_delta_refpoints()
    # if last_mean_delta_nadir is not None:
    #     tb_writer.add_scalar("Mean Delta Nadir", last_mean_delta_nadir, validator.epoch)
    #     tb_writer.add_scalar("Mean Delta Utopia", last_mean_delta_utopia, validator.epoch)

    # test
    
    # Create a linear gradient from light blue to dark blue
    
    ray_list, param_dict_list = generate_params(phn, 50, agent.device)
    logprobs_list, test_f_list, _ = solve_one_batch(agent, param_dict_list, test_batch, None)
    gradient = np.linspace(0,1,len(param_dict_list))
    colors = np.vstack((mcolors.to_rgba(light_blue), mcolors.to_rgba(dark_blue)))
    my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors, N=len(param_dict_list))
    
    plt.figure()
    plt.scatter(test_f_list[0,:,0], test_f_list[0,:,1], c=gradient, cmap=my_cmap)
    # test_batch = test_batch[1:]
    tb_writer.add_figure("Solutions "+args.test_instance_name+"-"+str(args.test_num_vehicles), plt.gcf(), validator.epoch)
    # test_batch2 = test_batch2[1:]
    logprobs_list, test_f_list, _ = solve_one_batch(agent, param_dict_list, test_batch2, None)
    plt.figure()
    plt.scatter(test_f_list[0,:,0], test_f_list[0,:,1], c=gradient, cmap=my_cmap)
    tb_writer.add_figure("Solutions bar-n400-1-"+str(args.test_num_vehicles), plt.gcf(), validator.epoch)
    validator.epoch +=1

def initialize(param, phn, opt, tb_writer):
    ray = np.asanyarray([[0.5,0.5]],dtype=float)
    ray = torch.from_numpy(ray).to(phn.device, dtype=torch.float32)
    param_dict = phn(ray)
    pcs_weight = (param_dict["pcs_weight"]).ravel()
    pns_weight = (param_dict["pns_weight"]).ravel()
    po_weight = (param_dict["po_weight"]).ravel()
    weights = torch.concatenate([pcs_weight,pns_weight,po_weight], dim=0)
    loss = torch.norm(weights-param)
    opt.zero_grad(set_to_none=True)
    tb_writer.add_scalar("Initialization loss", loss.cpu().item())
    loss.backward()
    opt.step()
    return loss.cpu().item()

def init_phn_output(agent, phn, tb_writer, max_step=1000):
    po_weight = None
    pcs_weight = None
    pns_weight = None
    for name, param in agent.named_parameters():
        if name == "project_out.weight":
            po_weight = param.data.ravel()
        if name == "project_current_vehicle_state.weight":
            pcs_weight = param.data.ravel()
        if name == 'project_node_state.weight':
            pns_weight = param.data.ravel()
        
    pcs_weight = pcs_weight.detach().clone()
    pns_weight = pns_weight.detach().clone()
    po_weight = po_weight.detach().clone()
    weights = torch.concatenate([pcs_weight, pns_weight, po_weight], dim=0)
    opt_init = torch.optim.AdamW(phn.parameters(), lr=1e-4)
    for i in range(max_step):
        loss = initialize(weights,phn,opt_init,tb_writer)
        if loss < 1e-4:
            break

def update_policy(policy_type:str, policy:Policy, sample_list, score_list):
    if policy_type == "r1-nes":
        # score_list = np.concatenate(score_list, axis=1)
        # score_list = np.mean(score_list, axis=1, keepdims=True)
        x_list = sample_list - policy.mu
        w_list = x_list/math.exp(policy.ld)
        policy.update(w_list, x_list, score_list)
    elif policy_type == "crfmnes":
        policy.update(sample_list, score_list)
    return policy

def save_policy(policy, epoch, title):
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(title+".pt")
    checkpoint = {
        "policy":policy,
        "epoch":epoch,
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())

def save_phn(phn, epoch, title):
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(title+".pt")
    checkpoint = {
        "phn_state_dict":phn.state_dict(),
        "epoch":epoch,
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())

