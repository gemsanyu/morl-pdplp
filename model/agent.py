import math
from typing import Optional, Dict, List

import numpy as np
import torch
from torch.nn import Linear

CPU_DEVICE = torch.device("cpu")

from model.graph_encoder import GraphAttentionEncoder

class Agent(torch.nn.Module):
    def __init__(self,
                 num_node_static_features:int,
                 num_vehicle_dynamic_features:int,
                 num_node_dynamic_features:int,
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
        self.val_size = self.embed_dim // self.n_heads
        self.key_size = self.val_size
        # embedder
        self.gae = GraphAttentionEncoder(n_heads=n_heads,
                                         n_layers=n_gae_layers,
                                         embed_dim=embed_dim,
                                         node_dim=None,
                                         feed_forward_hidden=gae_ff_hidden)
        self.depot_embedder = Linear(num_node_static_features, embed_dim)
        self.pick_embedder = Linear(2*num_node_static_features, embed_dim)
        self.delivery_embedder = Linear(num_node_static_features, embed_dim)
        
        self.project_embeddings = Linear(embed_dim, 3*embed_dim, bias=False)
        self.project_fixed_context = Linear(embed_dim, embed_dim, bias=False)
        current_state_dim = embed_dim + num_vehicle_dynamic_features
        self.project_current_state = Linear(current_state_dim, embed_dim, bias=False)
        self.project_node_state = Linear(num_node_dynamic_features, 3*embed_dim, bias=False)
        self.project_out = Linear(embed_dim, embed_dim, bias=False)
        self.to(self.device)
        
    def _make_heads(self, x: torch.Tensor)->torch.Tensor:
        x = x.unsqueeze(2).view(x.size(0), x.size(1), self.n_heads, self.key_size)
        x = x.permute(2,0,1,3)
        return x

    def forward(self,
                num_vehicles: List[np.ndarray],
                node_embeddings: torch.Tensor,
                fixed_context: torch.Tensor,
                prev_node_embeddings: torch.Tensor,
                node_dynamic_features: torch.Tensor,
                vehicle_dynamic_features: torch.Tensor,
                glimpse_V_static: torch.Tensor,
                glimpse_K_static: torch.Tensor,
                logit_K_static: torch.Tensor,
                feasibility_mask: torch.Tensor,
                param_dict: Optional[Dict[str, torch.Tensor]]=None
                ):
        batch_size, num_nodes, _ = node_embeddings.shape
        num_vehicles_cum = np.insert(np.cumsum(num_vehicles),0,0)
        total_num_vehicles = np.sum(num_vehicles)
        feasibility_mask = torch.cat(feasibility_mask)
        # prepare the static to be repeated as many as the number of vehicles
        # in each batch size
        glimpse_V_static_list = [glimpse_V_static[:, i].unsqueeze(1).expand((-1,num_vehicles[i],-1,-1)) for i in range(batch_size)]
        glimpse_V_static = torch.cat(glimpse_V_static_list, dim=1)
        glimpse_K_static_list = [glimpse_K_static[:, i].unsqueeze(1).expand((-1,num_vehicles[i],-1,-1)) for i in range(batch_size)]
        glimpse_K_static = torch.cat(glimpse_K_static_list, dim=1)
        logit_K_static_list = [logit_K_static[i,:].unsqueeze(0).expand((num_vehicles[i],-1,-1)) for i in range(batch_size)]
        logit_K_static = torch.cat(logit_K_static_list, dim=0)
        # now decode
        fixed_context = fixed_context.unsqueeze(1)  
        fixed_context_each_vehicle = [torch.cat([fixed_context[i].expand((num_vehicles[i],-1)),vehicle_dynamic_features[i]], dim=1) for i in range(batch_size)]
        fixed_context_each_vehicle = torch.cat([fixed_context_each_vehicle[i] for i in range(batch_size)], dim=0)
        fixed_context = torch.cat([fixed_context[i].unsqueeze(0).expand((num_vehicles[i],-1,-1)) for i in range(batch_size)])
        if param_dict is None:
            projected_current_state = self.project_current_state(fixed_context_each_vehicle).unsqueeze(1) 
            node_dynamic_embeddings = self.project_node_state(torch.cat([node_dynamic_features[i] for i in range(batch_size)], dim=0))
            glimpse_V_dynamic, glimpse_K_dynamic, logit_K_dynamic = node_dynamic_embeddings.chunk(3, dim=-1)
        glimpse_V_dynamic = self._make_heads(glimpse_V_dynamic)
        glimpse_K_dynamic = self._make_heads(glimpse_K_dynamic)
        glimpse_V = glimpse_V_static + glimpse_V_dynamic
        glimpse_K = glimpse_K_static + glimpse_K_dynamic
        logit_K = logit_K_static + logit_K_dynamic
        query = fixed_context + projected_current_state
        glimpse_Q = query.view(total_num_vehicles, self.n_heads, 1, self.key_size)
        glimpse_Q = glimpse_Q.permute(1,0,2,3)
        compatibility = glimpse_Q@glimpse_K.permute(0,1,3,2) / math.sqrt(glimpse_Q.size(-1)) # glimpse_K => n_heads, batch_size, num_items, embed_dim
        compatibility = compatibility + feasibility_mask.unsqueeze(0).unsqueeze(2).float().log()
        attention = torch.softmax(compatibility, dim=-1)
        heads = attention@glimpse_V

        concated_heads = heads.permute(1,2,0,3).contiguous()
        concated_heads = concated_heads.view(total_num_vehicles, 1, self.embed_dim)
        if param_dict is None:
            final_Q = self.project_out(concated_heads)
        logits = final_Q@logit_K.permute(0,2,1) / math.sqrt(final_Q.size(-1)) #batch_size, num_items, embed_dim
        logits = torch.tanh(logits) * self.tanh_clip
        logits = logits.squeeze(1) + feasibility_mask.float().log()
        # now we need to re-split the logits into their respective batch
        logits = [logits[num_vehicles_cum[i-1]:num_vehicles_cum[i]].flatten() for i in range(1,batch_size+1)]
        probs = [torch.softmax(logits[i],dim=-1) for i in range(batch_size)]
        
        select_result_list = [self.select(probs[i]) for i in range(batch_size)]
        action_list = [select_result_list[i][0] for i in range(batch_size)]
        logprob_list = [select_result_list[i][1] for i in range(batch_size)]
        entropy_list = [select_result_list[i][2] for i in range(batch_size)]
        selected_vec = [action_list[i]/num_nodes for i in range(batch_size)]
        selected_node = [action_list[i]%num_nodes for i in range(batch_size)] 
        return selected_vec, selected_node, logprob_list, entropy_list

    def select(self, probs):
        '''
        ### Select next to be executed.
        -----
        Parameter:
            probs: probabilities of each operation

        Return: index of operations, log of probabilities
        '''
        if self.training:
            dist = torch.distributions.Categorical(probs)
            op = dist.sample()
            logprob = dist.log_prob(op)
            entropy = dist.entropy()
        else:
            prob, op = torch.max(probs, dim=-1)
            logprob = torch.log(prob)
            entropy = -torch.sum(prob*logprob)
        return op, logprob, entropy