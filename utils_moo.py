import math
import pathlib

import numpy as np
import torch

from policy.policy import Policy

def update_policy(policy_type:str, policy:Policy, sample_list, score_list):
    if policy_type == "r1-nes":
        score_list = np.concatenate(score_list, axis=1)
        score_list = np.mean(score_list, axis=1, keepdims=True)
        x_list = sample_list - policy.mu
        w_list = x_list/math.exp(policy.ld)
        policy.update(w_list, x_list, score_list)
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

   
def save_validator(validator, title):
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(title+"_validator.pt")
    checkpoint = {
        "validator":validator,
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())

