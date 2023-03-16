import random

import numpy as np
import torch

from bpdplp.bpdplp_dataset import BPDPLP_Dataset
from utils_moo import prepare_args, validate_one_epoch
from validator import save_validator
from setup_phn import setup_phn

def run(args):
    agent, phn, opt, validator, tb_writer, test_batch, test_batch2, last_epoch = setup_phn(args, validation=True)
    validation_dataset = BPDPLP_Dataset(num_samples=args.num_validation_samples, mode="validation")
    validate_one_epoch(args, agent, phn, validator, validation_dataset, test_batch, test_batch2, tb_writer, validator.epoch)
    save_validator(validator, args.title)
        
if __name__ == "__main__":
    args = prepare_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)