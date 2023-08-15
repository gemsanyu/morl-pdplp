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
from model.agent_mo import Agent
from model.phn import PHN
from policy.hv import Hypervolume
from utils import prepare_args
from utils_moo import save_phn, generate_params, get_ray_list, solve_one_batch, compute_loss
from utils_moo import update_phn, init_phn_output, compute_spread_loss, init_one_epoch
from setup_phn import setup_phn

LIGHT_BLUE = mcolors.CSS4_COLORS['lightblue']
DARK_BLUE = mcolors.CSS4_COLORS['darkblue']

def plot_training_progress(tb_writer, epoch, hv_loss_list, spread_loss_list, cos_penalty_loss_list):
    tb_writer.add_scalar("Training HV LOSS", hv_loss_list.mean(), epoch)
    tb_writer.add_scalar("Training Spread Loss", spread_loss_list.mean(), epoch)
    tb_writer.add_scalar("Cos Penalty Loss", cos_penalty_loss_list.mean(), epoch)
    
def train_one_epoch(args, agent: Agent, phn: PHN, critic_phn: PHN, opt, train_dataset, training_nondom_list, tb_writer, epoch, init_stage=False):

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    ld = 10 if init_stage else args.ld 
    if training_nondom_list is None:
        training_nondom_list = [None for i in range(len(train_dataset))]
    cos_penalty_loss_list = []
    hv_loss_list = []
    spread_loss_list = []
    for _, batch in tqdm(enumerate(train_dataloader), desc=f'Training epoch {epoch}'):
        ray_list =  get_ray_list(args.num_ray, agent.device)
        # get solutions
        agent.train()
        param_dict_list = generate_params(phn, ray_list)
        logprob_list, batch_f_list, _, training_nondom_list = solve_one_batch(agent, param_dict_list, batch, training_nondom_list)
        # get baseline/critic
        agent.eval()
        with torch.no_grad():
            crit_param_dict_list = generate_params(critic_phn, ray_list)
            _, greedy_batch_f_list, _, training_nondom_list = solve_one_batch(agent, crit_param_dict_list, batch, training_nondom_list)
        idx_list = batch[0]
        hv_loss, cos_penalty_loss = compute_loss(logprob_list, training_nondom_list, idx_list, batch_f_list, greedy_batch_f_list, ray_list)
        spread_loss = compute_spread_loss(logprob_list, training_nondom_list, idx_list, batch_f_list)
        final_loss = hv_loss - 0.01*spread_loss
        if init_stage:
            final_loss = 0
        final_loss -= ld*cos_penalty_loss
        update_phn(agent, phn, opt, final_loss)
        hv_loss_list += [hv_loss.detach().cpu().numpy()]
        spread_loss_list += [spread_loss.detach().cpu().numpy()]
        cos_penalty_loss_list += [cos_penalty_loss.detach().cpu().numpy()]
    hv_loss_list = np.array(hv_loss_list)
    spread_loss_list = np.array(spread_loss_list)
    cos_penalty_loss_list = np.array(cos_penalty_loss_list)
    plot_training_progress(tb_writer, epoch, hv_loss_list, spread_loss_list, cos_penalty_loss_list)
    return training_nondom_list  
        
@torch.no_grad()        
def validate_one_epoch(args, agent, phn, critic_phn, validation_nondom_list, critic_solution_list, validation_dataset, test_batch, test_batch2, tb_writer, epoch):
    agent.eval()
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size)
    if validation_nondom_list is None:
        validation_nondom_list = [None for _ in range(len(validation_dataset))]
    
    ray_list =  get_ray_list(10, agent.device, is_random=False)        
    if critic_solution_list is None:
        critic_solution_list = []
        crit_param_dict_list = generate_params(critic_phn, ray_list)
        for _, batch in tqdm(enumerate(validation_dataloader), desc=f'Validation epoch {epoch}'):
            _, batch_f_list, _, validation_nondom_list = solve_one_batch(agent, crit_param_dict_list, batch, validation_nondom_list)
            critic_solution_list += [batch_f_list]
        critic_solution_list = np.concatenate(critic_solution_list, axis=0)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size)
        
    param_dict_list = generate_params(phn, ray_list)
    f_list = []
    for batch_idx, batch in tqdm(enumerate(validation_dataloader), desc=f'Validation epoch {epoch}'):
        _, batch_f_list, _, validation_nondom_list = solve_one_batch(agent, param_dict_list, batch, validation_nondom_list)
        f_list += [batch_f_list] 
    f_list = np.concatenate(f_list,axis=0)
    is_improving, critic_solution_list, critic_phn = compare_with_critic(phn, critic_phn, validation_nondom_list, f_list, critic_solution_list, epoch, tb_writer)
    
    # plot 1 or 2 from validation?
    gradient = np.linspace(0,1,len(param_dict_list))
    colors = np.vstack((mcolors.to_rgba(LIGHT_BLUE), mcolors.to_rgba(DARK_BLUE)))
    my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors, N=len(param_dict_list))
    plt.figure()
    plt.scatter(f_list[0,:,0], f_list[0,:,1], c=gradient, cmap=my_cmap)
    tb_writer.add_figure("Solutions Validation 1", plt.gcf(), epoch)
    
    plt.figure()
    plt.scatter(f_list[1,:,0], f_list[1,:,1], c=gradient, cmap=my_cmap)
    tb_writer.add_figure("Solutions Validation 2", plt.gcf(), epoch)
    validate_with_test(agent, phn, test_batch, test_batch2, tb_writer, epoch)
    return is_improving, validation_nondom_list, critic_solution_list, critic_phn

def compare_with_critic(phn, critic_phn, validation_nondom_list, f_list, critic_solution_list, epoch, tb_writer):
    hv_list = []
    crit_hv_list = []
    for i in range(len(f_list)):
        nondom_f = validation_nondom_list[i]
        utopia_points = np.min(nondom_f, axis=0, keepdims=True)
        nadir_points = np.max(nondom_f, axis=0, keepdims=True)
        diff = nadir_points-utopia_points
        diff[diff==0] = 1
        norm_agent_f = (f_list[i, :]-utopia_points)/diff
        norm_critic_f = (critic_solution_list[i,:]-utopia_points)/diff
        agent_hv = Hypervolume(np.array([1.1,1.1])).calc(norm_agent_f)
        critic_hv = Hypervolume(np.array([1.1,1.1])).calc(norm_critic_f)
        hv_list += [agent_hv]
        crit_hv_list += [critic_hv]
    hv_list = np.asanyarray(hv_list)
    crit_hv_list = np.asanyarray(crit_hv_list)
    res = wilcoxon(hv_list, crit_hv_list, alternative="greater")
    is_improving = res.pvalue < 0.05
    tb_writer.add_scalar("Validation pvalue", res.pvalue, epoch)
    if is_improving:
        critic_phn.load_state_dict(copy.deepcopy(phn.state_dict()))
        critic_solution_list = f_list
    return is_improving, critic_solution_list, critic_phn

def validate_with_test(agent, phn, test_batch, test_batch2, tb_writer, epoch):
    ray_list =  get_ray_list(50, agent.device)
    param_dict_list = generate_params(phn, ray_list)
    _, test_f_list, _, _ = solve_one_batch(agent, param_dict_list, test_batch, None)
    gradient = np.linspace(0,1,len(param_dict_list))
    colors = np.vstack((mcolors.to_rgba(LIGHT_BLUE), mcolors.to_rgba(DARK_BLUE)))
    my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors, N=len(param_dict_list))
    plt.figure()
    plt.scatter(test_f_list[0,:,0], test_f_list[0,:,1], c=gradient, cmap=my_cmap)
    tb_writer.add_figure("Solutions "+args.test_instance_name+"-"+str(args.test_num_vehicles), plt.gcf(), epoch)
    
    _, test_f_list,_, _ = solve_one_batch(agent, param_dict_list, test_batch2, None)
    plt.figure()
    plt.scatter(test_f_list[0,:,0], test_f_list[0,:,1], c=gradient, cmap=my_cmap)
    tb_writer.add_figure("Solutions bar-n400-1-"+str(args.test_num_vehicles), plt.gcf(), epoch)
    

def run(args):
    patience=10
    not_improving_count = 0
    agent, phn, critic_phn, training_nondom_list, validation_nondom_list, critic_solution_list, opt, tb_writer, test_batch, test_batch2, last_epoch = setup_phn(args)
    validation_dataset = BPDPLP_Dataset(num_samples=args.num_validation_samples, mode="validation")
    train_dataset = BPDPLP_Dataset(num_samples=args.num_training_samples, mode="training")
    # init_phn_output(agent, phn, tb_writer, max_step=1000)
    opt_init = torch.optim.AdamW(phn.parameters(), lr=1e-4)
    phn = init_one_epoch(args, agent, phn, opt_init,train_dataset)
    init_epoch = 1
    opt_directions = torch.optim.AdamW(phn.parameters(), lr=1e-4)
    for epoch in range(last_epoch+1, args.max_epoch):
        if epoch <= init_epoch:
            training_nondom_list = train_one_epoch(args, agent, phn, critic_phn, opt_directions, train_dataset, training_nondom_list, tb_writer, epoch, init_stage=True)
        else:
            training_nondom_list = train_one_epoch(args, agent, phn, critic_phn, opt, train_dataset, training_nondom_list, tb_writer, epoch, init_stage=False)
        is_improving, validation_nondom_list, critic_solution_list, critic_phn = validate_one_epoch(args, agent, phn, critic_phn, validation_nondom_list, critic_solution_list, validation_dataset, test_batch, test_batch2, tb_writer, epoch)
        if not is_improving:
            not_improving_count += 1
            if not_improving_count == patience:
                break
        else:
            not_improving_count = 0
        save_phn(args.title, epoch, phn, critic_phn, opt, training_nondom_list, validation_nondom_list, critic_solution_list)


if __name__ == "__main__":
    matplotlib.use('Agg')
    args = prepare_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)
