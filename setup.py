import os
import pathlib

from model.agent_so import Agent
import torch
from torch.utils.tensorboard import SummaryWriter


NUM_NODE_STATIC_FEATURES = 6
NUM_VEHICLE_DYNAMIC_FEATURES = 4
NUM_NODE_DYNAMIC_FEATURES = 1

def setup(args, load_best=False):
    agent = Agent(num_node_static_features=NUM_NODE_STATIC_FEATURES,
                  num_vehicle_dynamic_features=NUM_VEHICLE_DYNAMIC_FEATURES,
                  num_node_dynamic_features=NUM_NODE_DYNAMIC_FEATURES,
                  n_heads=args.n_heads,
                  n_gae_layers=args.n_gae_layers,
                  embed_dim=args.embed_dim,
                  gae_ff_hidden=args.gae_ff_hidden,
                  tanh_clip=args.tanh_clip,
                  device=args.device)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    summary_root = "runs"
    summary_dir = pathlib.Path(".")/summary_root
    model_summary_dir = summary_dir/args.title
    model_summary_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=model_summary_dir.absolute())

    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(args.title+".pt")
    if load_best:
        checkpoint_path = checkpoint_dir/(args.title+"_best.pt")
    
    checkpoint = None
    if os.path.isfile(checkpoint_path.absolute()):
        checkpoint = torch.load(checkpoint_path.absolute(), map_location=args.device)
    else:
        print("CHECKPOINT NOT FOUND! new run?")

    last_epoch = 0
    best_agent_state_dict = None
    best_validation_score = None
    if checkpoint is not None:
        agent.load_state_dict(checkpoint["agent_state_dict"])
        best_agent_state_dict = checkpoint["best_agent_state_dict"]
        best_validation_score = checkpoint["best_validation_score"]
        opt.load_state_dict(checkpoint["agent_opt_state_dict"])
        last_epoch = checkpoint["epoch"]    

    return agent, opt, best_agent_state_dict, best_validation_score, tb_writer, last_epoch
