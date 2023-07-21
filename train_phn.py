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
from utils_moo import update_phn, get_ray
from setup_phn import setup_phn

LIGHT_BLUE = mcolors.CSS4_COLORS['lightblue']
DARK_BLUE = mcolors.CSS4_COLORS['darkblue']

def train_one_epoch(args, agent: Agent, phn: PHN, opt, train_dataset, training_nondom_list, tb_writer, epoch):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    
    agent.train()
    for _, batch in tqdm(enumerate(train_dataloader), desc=f'Training epoch {epoch}'):
        ray = get_ray(agent.device)
        param_dict = phn(ray)
        logprobs, f_list, training_nondom_list = solve_one_batch(agent, param_dict, batch, training_nondom_list)
        idx_list = batch[0]
        loss = compute_loss(logprobs, training_nondom_list, idx_list, f_list, ray)
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
            _, _, training_nondom_list = solve_one_batch(agent, param_dict, batch, training_nondom_list)
    return training_nondom_list



def run(args):
    agent, phn, training_nondom_list, validation_nondom_list, opt, tb_writer, test_batch, test_batch2, last_epoch = setup_phn(args)
    validation_dataset = BPDPLP_Dataset(num_samples=args.num_validation_samples, mode="validation")
    train_dataset = BPDPLP_Dataset(num_samples=args.num_training_samples, mode="training")
    # population training nondom list if still None or first epoch
    if training_nondom_list is None:
        training_nondom_list = populate_nondom_list(agent, phn, train_dataset, args.batch_size)
    
    for epoch in range(last_epoch+1, args.max_epoch):
        training_nondom_list = train_one_epoch(args, agent, phn, opt, train_dataset, training_nondom_list, tb_writer, epoch)
        validation_nondom_list, critic_solution_list, critic_phn = validate_one_epoch(args, agent, phn, critic_phn, validation_nondom_list, critic_solution_list, validation_dataset, test_batch, test_batch2, tb_writer, epoch)
        # save_phn(args.title, epoch, phn, critic_phn, opt, training_nondom_list, validation_nondom_list, critic_solution_list)



if __name__ == "__main__":
    matplotlib.use('Agg')
    args = prepare_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)