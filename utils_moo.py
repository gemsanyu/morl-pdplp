import math
import pathlib

import numpy as np
import torch
import random

from torch.utils.data import DataLoader
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

def compute_loss(logprobs, training_nondom_list, idx_list, reward_list, ray):
    device = logprobs.device
    nadir = []
    utopia = []
    for i in range(len(reward_list)):
        nondom_sols =  training_nondom_list[idx_list[i]]
        nadir += [np.max(nondom_sols, axis=0, keepdims=True)]
        utopia += [np.min(nondom_sols, axis=0, keepdims=True)]
    nadir = np.concatenate(nadir, axis=0)[:,np.newaxis,:]
    utopia = np.concatenate(utopia, axis=0)[:,np.newaxis,:]
    
    denom = nadir-utopia
    denom[denom==0]=1
    reward_list *= -1
    reward_list = (reward_list-utopia)/denom
    reward_list = torch.from_numpy(reward_list).to(device)
    reward_list *= -1
    ray = ray[None, None, :]
    # print(reward_list)
    tch_reward = ray*(reward_list)
    tch_reward, _ = tch_reward.max(dim=-1)
    tch_advantage = tch_reward - tch_reward.mean(dim=1, keepdim=True)
    loss = tch_advantage*logprobs
    loss = -loss.mean()
    return loss

def update(agent, opt, loss):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
    opt.step()
    opt.zero_grad(set_to_none=True)

def update_step_only(agent, opt):
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
    opt.step()
    opt.zero_grad(set_to_none=True)

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

def solve_one_batch(agent, batch, nondom_list):
    idx_list = batch[0]
    batch = batch[1:]
    num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types = batch
    env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
    static_features,_,_,_ = env.begin()
    static_features = torch.from_numpy(static_features).to(agent.device)
    encode_results = encode(agent, static_features)
    node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_results

    solve_results = solve_decode_only(agent, env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static)
    tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties, reward_list, logprob_list, sum_entropies = solve_results
    f_list = np.concatenate([travel_costs[:,np.newaxis], late_penalties[:,np.newaxis]], axis=1)

    if nondom_list is None:
        return logprob_list, f_list, reward_list, None
    
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

    return logprob_list, f_list, reward_list, nondom_list

def save(title, epoch, agent, critic, opt, training_nondom_list, validation_nondom_list, critic_solution_list):
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(title+".pt")
    checkpoint = {
        "agent_state_dict": agent.state_dict(),
        "critic_state_dict": critic.state_dict(),
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

