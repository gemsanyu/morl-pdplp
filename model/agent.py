import torch
from torch.nn import Linear

CPU_DEVICE = torch.device("cpu")

from model.graph_encoder import GraphAttentionEncoder

class Agent(torch.nn.Module):
    def __init__(self,
                 static_features:int,
                 n_heads: int,
                 n_gae_layers: int,
                 embed_dim: int,
                 gae_ff_hidden: int,
                 tanh_clip: float,
                 device=CPU_DEVICE):
        super(Agent, self).__init__()
        self.n_heads = n_heads
        self.n_gae_layers = n_gae_layers
        self.embed_dim = embed_dim
        self.tanh_clip = tanh_clip
        self.device = device
        self.key_size = self.val_size = self.embed_dim // self.n_heads
        # embedder
        self.gae = GraphAttentionEncoder(n_heads=n_heads,
                                         n_layers=n_gae_layers,
                                         embed_dim=embed_dim,
                                         node_dim=None,
                                         feed_forward_hidden=gae_ff_hidden)
        self.depot_embedder = Linear(static_features, embed_dim)
        self.pick_embedder = Linear(2*static_features, embed_dim)
        self.delivery_embedder = Linear(static_features, embed_dim)
        self.to(self.device)