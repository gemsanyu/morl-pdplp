import random
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bpdplp.bpdplp_dataset import BPDPLP_Dataset
from model.agent import Agent
from model.phn import PHN
from utils_moo import save_phn, generate_params, solve_one_batch, compute_loss
from utils_moo import update_phn, init_phn_output, prepare_args, validate_one_epoch
from validator import save_validator
from setup_phn import setup_phn

def train_one_epoch(args, agent:Agent, phn:PHN, opt, train_dataset, validator, validation_dataset, test_batch, test_batch2, tb_writer, epoch, init_stage=False):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    ld = args.ld
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc=f'Training epoch {epoch}'):
        agent.train()
        ray_list, param_dict_list = generate_params(phn, args.num_ray, agent.device)
        logprob_list, batch_f_list = solve_one_batch(agent, param_dict_list, batch)
        agent.eval()
        with torch.no_grad():
            greedy_logprob_list, greedy_batch_f_list = solve_one_batch(agent, param_dict_list, batch)
        final_loss, total_cos_penalty = compute_loss(logprob_list, batch_f_list, greedy_batch_f_list, ray_list)
        if init_stage:
            final_loss = 0
            ld = 100
        final_loss -= ld*total_cos_penalty
        update_phn(agent, phn, opt, final_loss)
        
        if batch_idx % 5 == 0:
            tb_writer.add_scalar("COS PENALTY", total_cos_penalty.cpu().item())
            save_phn(phn, epoch, args.title)
            validate_one_epoch(args, agent, phn, validator, validation_dataset, test_batch, test_batch2, tb_writer, epoch)
            save_validator(validator, args.title)
            print("IMPROVING?",validator.is_improving)
            if not validator.is_improving:
                exit()
            
    save_phn(phn, epoch, args.title)        

def run(args):
    agent, phn, opt, validator, tb_writer, test_batch, test_batch2, last_epoch = setup_phn(args)
    validation_dataset = BPDPLP_Dataset(num_samples=args.num_validation_samples, mode="validation")
    train_dataset = BPDPLP_Dataset(num_samples=args.num_training_samples, mode="training")
    init_phn_output(agent, phn, tb_writer, max_step=1000)
    init_epoch = 5
    for epoch in range(last_epoch+1, args.max_epoch):
        if epoch < init_epoch:
            train_one_epoch(args, agent, phn, opt, train_dataset, validator, validation_dataset, test_batch, test_batch2, tb_writer, epoch, init_stage=True)
        else:
            train_one_epoch(args, agent, phn, opt, train_dataset, validator, validation_dataset, test_batch, test_batch2, tb_writer, epoch, init_stage=False)

        
if __name__ == "__main__":
    args = prepare_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)