import os
import pathlib

from model.agent import Agent
import torch
from torch.utils.tensorboard import SummaryWriter

def setup(args, load_best=False) -> Agent:
    agent = Agent(num_node_static_features=6,
                  num_vehicle_dynamic_features=4,
                  num_node_dynamic_features=1,
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
    if checkpoint is not None:
        agent.load_state_dict(checkpoint["state_dict"])
        opt.load_state_dict(checkpoint["opt_state_dict"])
        last_epoch = checkpoint["epoch"]    

    return agent, opt, tb_writer, last_epoch
