import copy
import random

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.colors as mcolors
from scipy.stats import wilcoxon

from bpdplp.bpdplp_dataset import BPDPLP_Dataset
from model.agent import Agent
from model.phn import PHN
from policy.hv import Hypervolume
from utils import prepare_args
from utils_moo import save_phn, get_ray_list, solve_one_batch, compute_loss
from utils_moo import update_phn, get_ray, generate_params
from setup_phn import setup_phn

LIGHT_BLUE = mcolors.CSS4_COLORS['lightblue']
DARK_BLUE = mcolors.CSS4_COLORS['darkblue']

def train_one_epoch(args, agent: Agent, phn: PHN, opt, train_dataset, training_nondom_list, tb_writer, epoch):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    
    agent.train()
    for _, batch in tqdm(enumerate(train_dataloader), desc=f'Training epoch {epoch}'):
        ray = get_ray(agent.device)
        param_dict = phn(ray)
        logprobs, f_list, reward_list, training_nondom_list = solve_one_batch(agent, param_dict, batch, training_nondom_list)
        idx_list = batch[0]
        loss = compute_loss(logprobs, training_nondom_list, idx_list, reward_list, ray)
        update_phn(agent, phn, opt, loss)
    return training_nondom_list  


@torch.no_grad()
def populate_nondom_list(agent, phn, train_dataset, batch_size, num_ray=10):
    training_nondom_list = [None for i in range(len(train_dataset))]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    agent.train() # random decoding
    ray_list = get_ray_list(num_ray, agent.device)
    param_dict_list = [phn(ray) for ray in ray_list]
    for _, batch in tqdm(enumerate(train_dataloader), desc="Populate Training Nondom List"):
        for param_dict in param_dict_list:
            _, _, _, training_nondom_list = solve_one_batch(agent, param_dict, batch, training_nondom_list)
    return training_nondom_list

@torch.no_grad()        
def validate_one_epoch(args, agent, critic, phn, critic_phn, validation_nondom_list, critic_solution_list, validation_dataset, test_batch, test_batch2, tb_writer, epoch):
    agent.eval()
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size)
    if validation_nondom_list is None:
        validation_nondom_list = [None for _ in range(len(validation_dataset))]
    
    ray_list =  get_ray_list(10, agent.device, is_random=False)        
    if critic_solution_list is None:
        critic_solution_list = []
        crit_param_dict_list = generate_params(critic_phn, ray_list)
        for _, batch in tqdm(enumerate(validation_dataloader), desc=f'Validation epoch {epoch}'):
            batch_f_list = []
            for crit_param_dict in crit_param_dict_list:
                _, f_list, _, validation_nondom_list = solve_one_batch(critic, crit_param_dict, batch, validation_nondom_list)
                batch_f_list += [f_list[:, np.newaxis, :]]
            batch_f_list = np.concatenate(batch_f_list, axis=1)
            critic_solution_list += [batch_f_list]
        critic_solution_list = np.concatenate(critic_solution_list, axis=0)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size)
    
    param_dict_list = generate_params(phn, ray_list)
    solution_list = []
    for batch_idx, batch in tqdm(enumerate(validation_dataloader), desc=f'Validation epoch {epoch}'):
        batch_f_list = []
        for param_dict in param_dict_list:
            _, f_list, _, validation_nondom_list = solve_one_batch(agent, param_dict, batch, validation_nondom_list)
            batch_f_list += [f_list[:, np.newaxis, :]]
        batch_f_list = np.concatenate(batch_f_list, axis=1)
        solution_list += [batch_f_list]
    solution_list = np.concatenate(solution_list,axis=0)
    critic_solution_list, critic, critic_phn = compare_with_critic(agent, critic, phn, critic_phn, validation_nondom_list, solution_list, critic_solution_list)
    
    # plot 1 or 2 from validation?
    gradient = np.linspace(0,1,len(param_dict_list))
    colors = np.vstack((mcolors.to_rgba(LIGHT_BLUE), mcolors.to_rgba(DARK_BLUE)))
    my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors, N=len(param_dict_list))
    plt.figure()
    plt.scatter(solution_list[0,:,0], solution_list[0,:,1], c=gradient, cmap=my_cmap)
    tb_writer.add_figure("Solutions Validation 1", plt.gcf(), epoch)
    
    plt.figure()
    plt.scatter(solution_list[-1,:,0], solution_list[-1,:,1], c=gradient, cmap=my_cmap)
    tb_writer.add_figure("Solutions Validation 2", plt.gcf(), epoch)
    validate_with_test(agent, phn, test_batch, test_batch2, tb_writer, epoch)
    return validation_nondom_list, critic_solution_list, critic, critic_phn

def compare_with_critic(agent, critic, phn, critic_phn, validation_nondom_list, solution_list, critic_solution_list):
    hv_list = []
    crit_hv_list = []
    for i in range(len(solution_list)):
        nondom_f = validation_nondom_list[i]
        utopia_points = np.min(nondom_f, axis=0, keepdims=True)
        nadir_points = np.max(nondom_f, axis=0, keepdims=True)
        diff = nadir_points-utopia_points
        diff[diff==0] = 1
        norm_agent_f = (solution_list[i, :]-utopia_points)/diff
        norm_critic_f = (critic_solution_list[i,:]-utopia_points)/diff
        agent_hv = Hypervolume(np.array([1.1,1.1])).calc(norm_agent_f)
        critic_hv = Hypervolume(np.array([1.1,1.1])).calc(norm_critic_f)
        hv_list += [agent_hv]
        crit_hv_list += [critic_hv]
    hv_list = np.asanyarray(hv_list)
    crit_hv_list = np.asanyarray(crit_hv_list)
    res = wilcoxon(hv_list, crit_hv_list, alternative="greater")
    is_improving = res.pvalue < 0.05
    print("Validation pvalue ------ ", res.pvalue)
    if is_improving:
        critic.load_state_dict(copy.deepcopy(agent.state_dict()))
        critic_phn.load_state_dict(copy.deepcopy(phn.state_dict()))
        critic_solution_list = solution_list
    return critic_solution_list, critic, critic_phn

def validate_with_test(agent, phn, test_batch, test_batch2, tb_writer, epoch):
    ray_list =  get_ray_list(50, agent.device)
    param_dict_list = generate_params(phn, ray_list)
    test_f_list = []
    for param_dict in param_dict_list:
        _, f_list, _, _ = solve_one_batch(agent, param_dict, test_batch, None)
        test_f_list += [f_list[:, np.newaxis, :]]
    test_f_list = np.concatenate(test_f_list, axis=1)

    gradient = np.linspace(0,1,len(param_dict_list))
    colors = np.vstack((mcolors.to_rgba(LIGHT_BLUE), mcolors.to_rgba(DARK_BLUE)))
    my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors, N=len(param_dict_list))
    plt.figure()
    plt.scatter(test_f_list[0,:,0], test_f_list[0,:,1], c=gradient, cmap=my_cmap)
    tb_writer.add_figure("Solutions "+args.test_instance_name+"-"+str(args.test_num_vehicles), plt.gcf(), epoch)
    
    test_f_list = []
    for param_dict in param_dict_list:
        _, f_list,_, _ = solve_one_batch(agent, param_dict, test_batch2, None)
        test_f_list += [f_list[:, np.newaxis, :]]
    test_f_list = np.concatenate(test_f_list, axis=1)

    plt.figure()
    plt.scatter(test_f_list[0,:,0], test_f_list[0,:,1], c=gradient, cmap=my_cmap)
    tb_writer.add_figure("Solutions bar-n400-1-"+str(args.test_num_vehicles), plt.gcf(), epoch)
    

def run(args):
    agent, critic, phn, critic_phn, training_nondom_list, validation_nondom_list, critic_solution_list, opt, tb_writer, test_batch, test_batch2, last_epoch = setup_phn(args)
    validation_dataset = BPDPLP_Dataset(num_samples=args.num_validation_samples, mode="validation")
    train_dataset = BPDPLP_Dataset(num_samples=args.num_training_samples, mode="training")
    # population training nondom list if still None or first epoch
    if training_nondom_list is None:
        training_nondom_list = populate_nondom_list(agent, phn, train_dataset, args.batch_size)
    
    for epoch in range(last_epoch+1, args.max_epoch):
        training_nondom_list = train_one_epoch(args, agent, phn, opt, train_dataset, training_nondom_list, tb_writer, epoch)
        validation_nondom_list, critic_solution_list, critic, critic_phn = validate_one_epoch(args, agent, critic, phn, critic_phn, validation_nondom_list, critic_solution_list, validation_dataset, test_batch, test_batch2, tb_writer, epoch)
        save_phn(args.title, epoch, agent, critic, phn, critic_phn, opt, training_nondom_list, validation_nondom_list, critic_solution_list)

if __name__ == "__main__":
    matplotlib.use('Agg')
    args = prepare_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)