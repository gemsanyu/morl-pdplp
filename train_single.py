import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from arguments import get_parser
from bpdplp.bpdplp_env import BPDPLP_Env
from bpdplp.bpdplp_dataset import BPDPLP_Dataset
from utils import encode, solve_decode_only, save
from setup import setup

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

@torch.no_grad()
def validate_one_epoch(args, agent, tb_writer, epoch):
    agent.eval()
    validation_dataset = BPDPLP_Dataset(num_samples=args.num_validation_samples, mode="validation")
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    sum_validation_travel_costs = 0
    sum_validation_penalties = 0
    sum_validation_entropies = 0
    sum_validation_logprobs = 0
    
    for batch_idx, batch in tqdm(enumerate(validation_dataloader)):
        num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types = batch
        env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
        static_features,_,_,_ = env.begin()
        static_features = torch.from_numpy(static_features).to(agent.device)
        encode_results = encode(agent, static_features)
        node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_results
        solve_results = solve_decode_only(agent, env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static)
        tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties, sum_logprobs, sum_entropies = solve_results
        sum_validation_travel_costs += travel_costs.sum()
        sum_validation_penalties += late_penalties.sum()
        sum_validation_entropies += sum_entropies.sum().cpu()
        sum_validation_logprobs += sum_logprobs.sum().cpu()
    mean_validation_travel_costs = sum_validation_travel_costs/args.num_validation_samples
    mean_penalties = sum_validation_penalties/args.num_validation_samples
    tb_writer.add_scalar("Validation Travel Costs "+args.title, mean_validation_travel_costs, epoch)
    tb_writer.add_scalar("Validation Late Penalties "+args.title, mean_penalties, epoch)
    tb_writer.add_scalar("Validation Entropies "+args.title, sum_validation_entropies/args.num_validation_samples, epoch)
    validation_score = mean_validation_travel_costs + mean_penalties
    return validation_score
        
def train_one_epoch(args, agent, opt, tb_writer, epoch):
    train_dataset = BPDPLP_Dataset(num_samples=args.num_training_samples, mode="training")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    sum_advantage = 0
    sum_training_travel_costs = 0
    sum_training_penalties = 0
    sum_greedy_travel_costs = 0
    sum_greedy_penalties = 0
    sum_training_entropies = 0
    sum_greedy_entropies = 0
    sum_greedy_logprobs = 0
    sum_training_logprobs = 0
    
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc=f'Training epoch {epoch}'):
        agent.train()
        num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types = batch
        env = BPDPLP_Env(num_vehicles, max_capacity, coords, norm_coords, demands, norm_demands, planning_time, time_windows, norm_time_windows, service_durations, norm_service_durations, distance_matrix, norm_distance_matrix, road_types)
        static_features,_,_,_ = env.begin()
        static_features = torch.from_numpy(static_features).to(agent.device)
        encode_results = encode(agent, static_features)
        node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_results
        solve_results = solve_decode_only(agent, env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static)
        
        tour_list, arrived_time_list, departure_time_list, travel_costs, late_penalties, sum_logprobs, sum_entropies = solve_results
        obj_list = travel_costs + late_penalties
        agent.eval()
        with torch.no_grad():
            env.reset()
            solve_results_greedy = solve_decode_only(agent, env, node_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static)
            greedy_tour_list, greedy_arrived_time_list, greedy_departure_time_list, greedy_travel_costs, greedy_late_penalties, greedy_sum_logprobs, greedy_sum_entropies = solve_results_greedy
            greedy_obj_list = greedy_travel_costs + greedy_late_penalties
        # remember that the optimizer wants to minimize the loss
        # so adv = R - baseline
        # if R is smaller than baseline, that means it's better
        advantage = obj_list - greedy_obj_list
        advantage = torch.from_numpy(advantage).to(agent.device)
        loss = sum_logprobs*advantage + 0.05*sum_entropies
        loss = loss.mean()
        
        # update
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=args.max_grad_norm)
        opt.step()
        
        sum_advantage += advantage.sum().detach().cpu()
        sum_training_travel_costs += travel_costs.sum()
        sum_training_penalties += late_penalties.sum()
        sum_greedy_travel_costs += greedy_travel_costs.sum()
        sum_greedy_penalties += greedy_late_penalties.sum()
        sum_training_entropies += sum_entropies.sum().detach().cpu()
        sum_greedy_entropies += greedy_sum_entropies.sum().detach().cpu()
        sum_greedy_logprobs += greedy_sum_logprobs.sum().detach().cpu()
        sum_training_logprobs += sum_logprobs.sum().detach().cpu()
    
    tb_writer.add_scalar("Training Advantage "+args.title, sum_advantage/args.num_training_samples, epoch)
    tb_writer.add_scalar("Training Travel Costs "+args.title, sum_training_travel_costs/args.num_training_samples, epoch)
    tb_writer.add_scalar("Training Late Penalties "+args.title, sum_training_penalties/args.num_training_samples, epoch)
    tb_writer.add_scalar("Training Entropies "+args.title, sum_training_entropies/args.num_training_samples, epoch)
        
def run(args):
    agent, opt, tb_writer, last_epoch = setup(args)
    for epoch in range(last_epoch+1, args.max_epoch):
        train_one_epoch(args, agent, opt, tb_writer, epoch)
        validation_score = validate_one_epoch(args, agent, tb_writer, epoch)
        save(agent, opt, validation_score, epoch, args.title)

if __name__ == "__main__":
    args = prepare_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)