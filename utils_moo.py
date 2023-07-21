import math
import pathlib

import numpy as np
import torch
import random

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

def compute_loss(logprobs, training_nondom_list, idx_list, f_list, ray):
    device = logprobs.device
    nadir = []
    utopia = []
    for i in range(len(f_list)):
        nondom_sols =  training_nondom_list[idx_list[i]]
        nadir += [np.max(nondom_sols, axis=0, keepdims=True)]
        utopia += [np.min(nondom_sols, axis=0, keepdims=True)]
    nadir = np.concatenate(nadir, axis=0)
    utopia = np.concatenate(utopia, axis=0)
    denom = nadir-utopia
    denom[denom==0]=1e-8
    norm_obj = (f_list-utopia)/denom
    norm_obj = torch.from_numpy(norm_obj).to(device)
    ray = ray.unsqueeze(0)
    tch_reward = ray*(norm_obj)
    tch_reward, _ = tch_reward.max(dim=-1)
    # tch_advantage = tch_reward - tch_reward.mean()
    loss = tch_reward*logprobs
    loss = loss.mean()
    return loss

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
    torch.nn.utils.clip_grad_norm_(phn.parameters(), max_norm=0.5)
    opt.step()

def get_ray_list(num_ray, device, is_random=True):
    ray_list = []
    for i in range(num_ray):
        if is_random:
            start, end = 0.1, np.pi/2-0.1
            r = np.random.uniform(start + i*(end-start)/num_ray, start+ (i+1)*(end-start)/num_ray)
            ray = np.array([np.cos(r),np.sin(r)], dtype='float32')
            ray /= ray.sum()
            ray *= np.random.randint(1, 5)*abs(np.random.normal(1, 0.2))
        else:
            ray = np.asanyarray([float(i)/float(num_ray-1), float(num_ray-1-i)/float(num_ray-1)], dtype=float)
        ray = torch.from_numpy(ray).to(device, dtype=torch.float32)
        ray_list += [ray]
    ray_list = torch.stack(ray_list)
    return ray_list

def get_ray(device):
    r = random.random()
    ray = np.array([r,1-r], np.float32)
    ray = torch.from_numpy(ray).to(device)
    return ray

def generate_params(phn, ray_list):
    param_dict_list = []
    for ray in ray_list:
        param_dict = phn(ray)
        param_dict_list += [param_dict]
    return param_dict_list

def solve_one_batch(agent, param_dict, batch, nondom_list):
    idx_list = batch[0]
    batch = batch[1:]
    num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types = batch
    env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
    static_features,_,_,_ = env.begin()
    static_features = torch.from_numpy(static_features).to(agent.device)
    encode_results = encode(agent, static_features, param_dict)
    node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_results

    solve_results = solve_decode_only(agent, env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
    tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties, sum_logprobs, sum_entropies = solve_results
    f_list = np.concatenate([travel_costs[:,np.newaxis], late_penalties[:,np.newaxis]], axis=1)

    if nondom_list is None:
        return sum_logprobs, f_list, None
    
    for i in range(env.batch_size):
        idx = idx_list[i]
        if nondom_list[idx] is None:
            nondom = f_list[i, np.newaxis, :]
            nondom_list[idx] = nondom
        else:
            nondom_old = nondom_list[idx]
            nondom_old = np.concatenate([nondom_old, f_list[i, np.newaxis, :]])
            I = fast_non_dominated_sort(nondom_old)[0]
            nondom_list[idx] = nondom_old[I]

    return sum_logprobs, f_list, nondom_list



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

def save_phn(title, epoch, phn, critic_phn, opt, training_nondom_list, validation_nondom_list, critic_solution_list):
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(title+".pt")
    checkpoint = {
        "phn_state_dict":phn.state_dict(),
        "critic_phn_state_dict":critic_phn.state_dict(),
        "opt_state_dict":opt.state_dict(),
        "training_nondom_list":training_nondom_list,
        "validation_nondom_list":validation_nondom_list,
        "critic_solution_list":critic_solution_list,
        "epoch":epoch,
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())

