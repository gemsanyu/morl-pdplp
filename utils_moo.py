import math
import pathlib

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
    # A = batch_f_list
    denom = (nadir-utopia)
    denom[denom==0] = 1
    norm_obj = (A-utopia)/denom
    # norm_obj = batch_f_list

    logprob_list = logprob_list.unsqueeze(2)
    loss_per_obj = logprob_list*torch.from_numpy(norm_obj).to(logprob_list.device)
    hv_d_list = get_hv_d(loss_per_obj.detach().cpu().numpy())
    hv_d_list = hv_d_list.to(device)
    hv_loss_per_obj = loss_per_obj*hv_d_list
    hv_loss_per_instance = hv_loss_per_obj.sum(dim=2)
    hv_loss_per_ray = hv_loss_per_instance.mean(dim=0)
    hv_loss = hv_loss_per_ray.sum()
    loss_min, _ = hv_loss_per_obj.min(dim=1, keepdim=True)
    loss_max, _ = hv_loss_per_obj.max(dim=1, keepdim=True)
    hv_loss_per_obj_norm = (hv_loss_per_obj-loss_min)/loss_max
    ray_list = ray_list.unsqueeze(0).expand_as(hv_loss_per_obj_norm)
    cos_penalty = cosine_similarity(hv_loss_per_obj_norm, ray_list, dim=2)*logprob_list.squeeze(-1)
    cos_penalty_per_ray = cos_penalty.mean(dim=0)
    total_cos_penalty = cos_penalty_per_ray.sum()
    return hv_loss, total_cos_penalty

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
    torch.nn.utils.clip_grad_norm_(phn.parameters(), max_norm=0.5)
    opt.step()
    agent.zero_grad(set_to_none=True)
    phn.zero_grad(set_to_none=True)
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

def generate_params(phn, ray_list):
    param_dict_list = []
    for ray in ray_list:
        param_dict = phn(ray)
        param_dict_list += [param_dict]
    return param_dict_list

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
        tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties, reward_list, logprobs, sum_entropies = solve_results
        sum_logprobs = logprobs.sum(dim=-1)
        f_list = np.concatenate([travel_costs[:,np.newaxis,np.newaxis], late_penalties[:,np.newaxis,np.newaxis]], axis=2)
        batch_f_list += [f_list]
        logprob_list += [sum_logprobs.unsqueeze(1)]
    logprob_list = torch.cat(logprob_list, dim=1)
    batch_f_list = np.concatenate(batch_f_list, axis=1)
    if nondom_list is None:
        return logprob_list, batch_f_list, reward_list, None
    
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

    return logprob_list, batch_f_list, reward_list, nondom_list



def initialize(param, phn, opt, tb_writer):
    ray = np.asanyarray([[0.5,0.5]],dtype=float)
    ray = torch.from_numpy(ray).to(phn.device, dtype=torch.float32)
    param_dict = phn(ray)
    weights = []
    weights += [(param_dict["pe_weight"]).ravel()] 
    weights += [(param_dict["pf_weight"]).ravel()]
    weights += [(param_dict["pcs_weight"]).ravel()] 
    weights += [(param_dict["pns_weight"]).ravel()]
    weights += [(param_dict["po_weight"]).ravel()]
    weights = torch.concatenate(weights, dim=0)
    loss = torch.norm(weights-param)
    opt.zero_grad(set_to_none=True)
    tb_writer.add_scalar("Initialization loss", loss.cpu().item())
    loss.backward()
    opt.step()
    return loss.cpu().item()

def init_phn_output(agent, phn, tb_writer, max_step=1000):
    pe_weight = None
    pf_weight = None
    pcs_weight = None
    pns_weight = None
    po_weight = None
    for name, param in agent.named_parameters():
        if name == "project_embeddings.weight":
            pe_weight = param.data.ravel()
        if name == "project_fixed_context.weight":
            pf_weight = param.data.ravel()
        if name == "project_current_vehicle_state.weight":
            pcs_weight = param.data.ravel()
        if name == 'project_node_state.weight':
            pns_weight = param.data.ravel()
        if name == "project_out.weight":
            po_weight = param.data.ravel()
        
        
    weights = []
    weights += [pe_weight.detach().clone()]
    weights += [pf_weight.detach().clone()]
    weights += [pcs_weight.detach().clone()]
    weights += [pns_weight.detach().clone()]
    weights += [po_weight.detach().clone()]
    weights = torch.concatenate(weights, dim=0)
    opt_init = torch.optim.AdamW(phn.parameters(), lr=1e-4)
    for i in range(max_step):
        loss = initialize(weights,phn,opt_init,tb_writer)
        if loss < 1e-4:
            break

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

