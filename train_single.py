import copy
import random

import numpy as np
import numba
from scipy.stats import wilcoxon
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from arguments import get_parser
from bpdplp.bpdplp_env import BPDPLP_Env
from bpdplp.bpdplp_dataset import BPDPLP_Dataset
from utils import encode, solve_decode_only, update_step_only
from utils import save, prepare_args
from setup import setup

def train_one_epoch(args, agent, best_agent, opt, train_dataset, tb_writer, epoch):
    agent.train()
    best_agent.eval()
    actual_batch_size = 128
    propagated_batch_size = 0
    train_dataloader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    sum_advantage = 0
    sum_training_travel_costs = 0
    sum_training_penalties = 0
    sum_training_entropies = 0
    
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc=f'Training epoch {epoch}'):
        _, num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types = batch
        env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
        static_features,_,_,_ = env.begin()
        static_features = torch.from_numpy(static_features).to(agent.device)
        encode_results = encode(agent, static_features)
        node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_results
        solve_results = solve_decode_only(agent, env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static)
        
        tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties, reward_list, logprob_list, sum_entropies = solve_results
        
        # computing critic
        with torch.no_grad():
            encode_results = encode(best_agent, static_features)
            node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_results
            greedy_solve_results = solve_decode_only(best_agent, env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static)    
            _, _, _, greedy_travel_costs, greedy_late_penalties, _, _, _ = greedy_solve_results

        score = travel_costs + late_penalties
        greedy_score = greedy_travel_costs + greedy_late_penalties
        tc_adv = travel_costs-greedy_travel_costs
        lp_adv = late_penalties-greedy_late_penalties
        tc_adv = (tc_adv-tc_adv.mean())/tc_adv.std()
        lp_adv = (lp_adv-lp_adv.mean())/lp_adv.std()
        advantage_list = tc_adv + lp_adv       
        logprob_list = logprob_list.sum(dim=-1)
        loss = logprob_list*torch.from_numpy(advantage_list).to(agent.device)
        loss = loss.mean() - 0.05*sum_entropies.mean()
        loss.backward()
        propagated_batch_size += actual_batch_size
        if propagated_batch_size == args.batch_size:
            propagated_batch_size = 0
            update_step_only(agent, opt, args.max_grad_norm)
        
        sum_advantage += advantage_list.sum()
        sum_training_travel_costs += travel_costs.sum()
        sum_training_penalties += late_penalties.sum()
        sum_training_entropies += sum_entropies.sum().detach().cpu()
    
    tb_writer.add_scalar("Training Advantage", sum_advantage/args.num_training_samples, epoch)
    tb_writer.add_scalar("Training Travel Costs", sum_training_travel_costs/args.num_training_samples, epoch)
    tb_writer.add_scalar("Training Late Penalties", sum_training_penalties/args.num_training_samples, epoch)
    tb_writer.add_scalar("Training Score", (sum_training_penalties+sum_training_travel_costs)/args.num_training_samples, epoch)
    tb_writer.add_scalar("Training Entropies", sum_training_entropies/args.num_training_samples, epoch)
        
@torch.no_grad()
def validate_one_epoch(agent, validation_dataset, best_validation_score, best_agent, tb_writer, epoch):
    agent.eval()
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True)
    validation_late_penalty_list = []
    validation_travel_cost_list = []

    for batch_idx, batch in tqdm(enumerate(validation_dataloader), desc=f'Validation epoch {epoch}'):
        _, num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types = batch
        env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
        static_features,_,_,_ = env.begin()
        static_features = torch.from_numpy(static_features).to(agent.device)
        encode_results = encode(agent, static_features)
        node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_results
        solve_results = solve_decode_only(agent, env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static)
        
        tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties, reward_list, logprob_list, sum_entropies = solve_results
        # advantage_list = (reward_list - reward_list.mean(axis=1, keepdims=True))/reward_list.std(axis=1, keepdims=True)
        # advantage_list = np.sum(advantage_list, axis=-1)
        
        validation_late_penalty_list += [late_penalties]
        validation_travel_cost_list += [travel_costs]
        # sum_advantage += advantage_list.sum()
        # sum_training_travel_costs += travel_costs.sum()
        # sum_training_penalties += late_penalties.sum()
        # sum_training_entropies += sum_entropies.sum().detach().cpu()

    validation_late_penalty_list = np.concatenate(validation_late_penalty_list)
    validation_travel_cost_list = np.concatenate(validation_travel_cost_list)

    # log the late penalties
    tb_writer.add_scalar("Validation Late Penalties", validation_late_penalty_list.mean(), epoch)
    tb_writer.add_scalar("Validation Travel Costs", validation_travel_cost_list.mean(), epoch)

    validation_score_list = validation_late_penalty_list + validation_travel_cost_list
    tb_writer.add_scalar("Validation Score", validation_score_list.mean(), epoch)

    if best_validation_score is None:
        best_validation_score = validation_score_list
        best_agent = copy.deepcopy(agent)
        return True, best_validation_score, best_agent
    ret = wilcoxon(validation_score_list, best_validation_score, alternative='less')
    tb_writer.add_scalar("pvalue", ret.pvalue, epoch)
    is_improving = ret.pvalue < 0.05
    if is_improving:
        best_validation_score = validation_score_list
        best_agent = copy.deepcopy(agent)
    return is_improving, best_validation_score, best_agent

def run(args):
    max_patience = 10
    not_improving = 0
    agent, opt, best_agent_state_dict, best_validation_score, tb_writer, last_epoch = setup(args)
    best_agent = copy.deepcopy(agent)
    if best_agent_state_dict is not None:
        best_agent.load_state_dict(best_agent_state_dict)
    validation_dataset = BPDPLP_Dataset(num_samples=args.num_validation_samples, mode="validation")
    train_dataset = BPDPLP_Dataset(num_samples=args.num_training_samples, mode="training")
    for epoch in range(last_epoch+1, args.max_epoch):
        train_one_epoch(args, agent, best_agent, opt, train_dataset, tb_writer, epoch)
        is_improving, best_validation_score, best_agent = validate_one_epoch(agent, validation_dataset, best_validation_score, best_agent, tb_writer, epoch)
        save(agent, opt, best_validation_score, best_agent.state_dict(), epoch, args.title)
        if not is_improving:
            not_improving += 1
        else:
            not_improving = 0
        if not_improving == max_patience:
            break

if __name__ == "__main__":
    args = prepare_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    numba.set_num_threads(4)
    run(args)