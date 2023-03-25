import os
import pathlib

import torch
from torch.utils.tensorboard import SummaryWriter

from bpdplp.bpdplp import BPDPLP
from model.agent import Agent
from policy.r1_nes import R1_NES
from policy.crfmnes import CRFMNES
from policy.policy import Policy
from validator import load_validator
from utils import instance_to_batch

CPU_DEVICE = torch.device("cpu")
POLICY_TYPE_DICT = {"r1-nes":R1_NES,
                    "crfmnes":CRFMNES}

NUM_NODE_STATIC_FEATURES = 4
NUM_VEHICLE_DYNAMIC_FEATURES = 2
NUM_NODE_DYNAMIC_FEATURES = 1

def get_agent(args, agent_checkpoint_path:pathlib.Path) -> Agent:
    agent = Agent(num_node_static_features=NUM_NODE_STATIC_FEATURES,
                  num_vehicle_dynamic_features=NUM_VEHICLE_DYNAMIC_FEATURES,
                  num_node_dynamic_features=NUM_NODE_DYNAMIC_FEATURES,
                  n_heads=args.n_heads,
                  n_gae_layers=args.n_gae_layers,
                  embed_dim=args.embed_dim,
                  gae_ff_hidden=args.gae_ff_hidden,
                  tanh_clip=args.tanh_clip,
                  device=args.device)

    checkpoint = torch.load(agent_checkpoint_path.absolute(), map_location=CPU_DEVICE)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    print(checkpoint["epoch"],"LAST AGENT EPOCH")
    return agent
    
def get_tb_writer(args)->SummaryWriter:
    summary_root = "runs"
    summary_dir = pathlib.Path(".")/summary_root
    model_summary_dir = summary_dir/args.title
    model_summary_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=model_summary_dir.absolute())
    return tb_writer

def get_policy(args, num_neurons) -> Policy :
    policy_class = POLICY_TYPE_DICT[args.policy]
    if args.policy == "r1-nes":
        policy = policy_class(num_neurons,NUM_VEHICLE_DYNAMIC_FEATURES, NUM_NODE_DYNAMIC_FEATURES, args.ld, args.negative_hv, args.lr, args.pop_size)
    elif args.policy == "crfmnes":
        policy = policy_class(num_neurons,NUM_VEHICLE_DYNAMIC_FEATURES, NUM_NODE_DYNAMIC_FEATURES, args.sigma, args.negative_hv)
    return policy

    
def setup_r1nes(args, load_best=False):
    tb_writer = get_tb_writer(args)    

    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent_checkpoint_path = checkpoint_dir/(args.title+"_agent.pt")
    agent = get_agent(args, agent_checkpoint_path)
    
    checkpoint_path = checkpoint_dir/(args.title+".pt")
    if load_best:
        checkpoint_path = checkpoint_dir/(args.title+"_best.pt")
    
    checkpoint = None
    if os.path.isfile(checkpoint_path.absolute()):
        checkpoint = torch.load(checkpoint_path.absolute(), map_location=args.device)
    else:
        print("CHECKPOINT NOT FOUND! new run?")

    policy = get_policy(args, args.embed_dim)
    policy.copy_to_mean(agent)
    validator = load_validator(args)
    last_epoch = 0
    if checkpoint is not None:
        policy = checkpoint["policy"]
        last_epoch = checkpoint["epoch"] 
        
    test_instance = BPDPLP(instance_name=args.test_instance_name,num_vehicles=args.test_num_vehicles)
    test_batch = instance_to_batch(test_instance)
    test_instance2 = BPDPLP(instance_name="bar-n400-1",num_vehicles=3)
    test_batch2 = instance_to_batch(test_instance2)
    return agent, policy, validator, tb_writer, test_batch, test_batch2, last_epoch
