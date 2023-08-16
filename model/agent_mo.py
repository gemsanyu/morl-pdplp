import math
from typing import Optional, Dict, List, Tuple

import torch
from torch.nn import Linear
import torch.nn.functional as F

from model.graph_encoder import GraphAttentionEncoder
from model.categorical import Categorical

CPU_DEVICE = torch.device("cpu")


# class Agent(torch.jit.ScriptModule):
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
        self.spatial_other_gae = GraphAttentionEncoder(n_heads=n_heads,
                                         n_layers=n_gae_layers,
                                         embed_dim=embed_dim,
                                         node_dim=None,
                                         feed_forward_hidden=gae_ff_hidden)
        self.depot_spatial_other_embedder = Linear(num_node_static_features-2, embed_dim)
        self.pick_spatial_other_embedder = Linear(2*num_node_static_features-4, embed_dim)
        self.delivery_spatial_other_embedder = Linear(num_node_static_features-2, embed_dim)
        
        self.temporal_gae = GraphAttentionEncoder(n_heads=n_heads,
                                         n_layers=n_gae_layers,
                                         embed_dim=embed_dim,
                                         node_dim=2*embed_dim,
                                         feed_forward_hidden=gae_ff_hidden)
        self.depot_temporal_embedder = Linear(2, embed_dim)
        self.pick_temporal_embedder = Linear(4, embed_dim)
        self.delivery_temporal_embedder = Linear(2, embed_dim)
        
        self.project_embeddings = Linear(embed_dim, 3*embed_dim, bias=False)
        self.project_fixed_context = Linear(embed_dim, embed_dim, bias=False)
        current_state_dim = embed_dim + num_vehicle_dynamic_features
        self.project_current_vehicle_state = Linear(current_state_dim, embed_dim, bias=False)
        self.project_node_state = Linear(num_node_dynamic_features, 3*embed_dim, bias=False)
        self.project_out = Linear(embed_dim, embed_dim, bias=False)
        self.to(self.device)
        
    # @torch.jit.script_method
    def _make_heads(self, x: torch.Tensor)->torch.Tensor:
        x = x.unsqueeze(2).view(x.size(0), x.size(1), self.n_heads, self.key_size)
        x = x.permute(2,0,1,3)
        return x
    
    # @torch.jit.script_method
    def forward(self,
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
        _, num_vehicles, _ = vehicle_dynamic_features.shape
        n_heads, key_size = self.n_heads, self.key_size
        current_vehicle_state = torch.cat([prev_node_embeddings, vehicle_dynamic_features], dim=-1)
        # if param_dict is not None:       
        #     projected_current_vehicle_state = F.linear(current_vehicle_state, param_dict["pcs_weight"])
        #     node_dynamic_embeddings = F.linear(node_dynamic_features,param_dict["pns_weight"])
        #     glimpse_V_dynamic, glimpse_K_dynamic, logit_K_dynamic = node_dynamic_embeddings.chunk(3, dim=-1)
        # else:
        projected_current_vehicle_state = self.project_current_vehicle_state(current_vehicle_state)
        node_dynamic_embeddings = self.project_node_state(node_dynamic_features)
        glimpse_V_dynamic, glimpse_K_dynamic, logit_K_dynamic = node_dynamic_embeddings.chunk(3, dim=-1)
        glimpse_V_dynamic = glimpse_V_dynamic.view((batch_size*num_vehicles,num_nodes,-1))
        glimpse_V_dynamic = self._make_heads(glimpse_V_dynamic)
        glimpse_V_dynamic = glimpse_V_dynamic.view((n_heads, batch_size, num_vehicles, num_nodes, -1))
        glimpse_K_dynamic = glimpse_K_dynamic.view((batch_size*num_vehicles,num_nodes,-1))
        glimpse_K_dynamic = self._make_heads(glimpse_K_dynamic)
        glimpse_K_dynamic = glimpse_K_dynamic.view((n_heads, batch_size, num_vehicles, num_nodes, -1))
        
        glimpse_V = glimpse_V_static + glimpse_V_dynamic
        glimpse_K = glimpse_K_static + glimpse_K_dynamic
        logit_K = logit_K_static + logit_K_dynamic
        query = fixed_context + projected_current_vehicle_state
        glimpse_Q = query.view(batch_size, num_vehicles, n_heads, 1, key_size)
        glimpse_Q = glimpse_Q.permute(2,0,1,3,4).contiguous()
        compatibility = glimpse_Q@glimpse_K.permute(0,1,2,4,3).contiguous() / math.sqrt(glimpse_Q.size(-1)) # glimpse_K => n_heads, batch_size, num_items, embed_dim
        compatibility = compatibility + feasibility_mask.unsqueeze(0).unsqueeze(3).float().log()
        
        # ini compatibility per vehicle untuk semua nodes
        # tapi attention sebenarnya harus per batch
        # jadi harus diflatten per batch, agar semua vec x customer jadi satu untuk disoftmax
        attention = torch.softmax(compatibility.view(n_heads,batch_size,num_vehicles*num_nodes),dim=-1).view_as(compatibility)
        heads = attention@glimpse_V
        # harus check bener gak yang kayak gini
        # kita concat vehicles across batch
        # sama kita isolasi per batch, tapi vehicle x nodees diflatten
        # kalau sama harusnya bener     
        concated_heads = heads.permute(1,2,3,0,4).contiguous()
        concated_heads = concated_heads.view(batch_size, num_vehicles, 1, self.embed_dim)
        # if param_dict is not None:
        final_Q = F.linear(concated_heads, param_dict["po_weight"])
        # else:
        #     final_Q = self.project_out(concated_heads)
        logits = final_Q@logit_K.permute(0,1,3,2) / math.sqrt(final_Q.size(-1)) #batch_size, num_items, embed_dim
        logits = torch.tanh(logits) * self.tanh_clip
        logits = logits.squeeze(2) + feasibility_mask.float().log()
        logits = logits.view(batch_size, num_vehicles*num_nodes)
        probs = torch.softmax(logits, dim=1)
        op, logprob_list, entropy_list = self.select(probs)
        selected_vecs = torch.floor(op/num_nodes).to(dtype=torch.long)
        selected_nodes = op % num_nodes
        return selected_vecs, selected_nodes, logprob_list, entropy_list

    def get_probs(self,
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
        _, num_vehicles, _ = vehicle_dynamic_features.shape
        n_heads, key_size = self.n_heads, self.key_size
        current_vehicle_state = torch.cat([prev_node_embeddings, vehicle_dynamic_features], dim=-1)
        # if param_dict is not None:       
        # projected_current_vehicle_state = F.linear(current_vehicle_state, param_dict["pcs_weight"])
        # node_dynamic_embeddings = F.linear(node_dynamic_features,param_dict["pns_weight"])
        # glimpse_V_dynamic, glimpse_K_dynamic, logit_K_dynamic = node_dynamic_embeddings.chunk(3, dim=-1)
        # else:
        projected_current_vehicle_state = self.project_current_vehicle_state(current_vehicle_state)
        node_dynamic_embeddings = self.project_node_state(node_dynamic_features)
        glimpse_V_dynamic, glimpse_K_dynamic, logit_K_dynamic = node_dynamic_embeddings.chunk(3, dim=-1)
        glimpse_V_dynamic = glimpse_V_dynamic.view((batch_size*num_vehicles,num_nodes,-1))
        glimpse_V_dynamic = self._make_heads(glimpse_V_dynamic)
        glimpse_V_dynamic = glimpse_V_dynamic.view((n_heads, batch_size, num_vehicles, num_nodes, -1))
        glimpse_K_dynamic = glimpse_K_dynamic.view((batch_size*num_vehicles,num_nodes,-1))
        glimpse_K_dynamic = self._make_heads(glimpse_K_dynamic)
        glimpse_K_dynamic = glimpse_K_dynamic.view((n_heads, batch_size, num_vehicles, num_nodes, -1))
        
        glimpse_V = glimpse_V_static + glimpse_V_dynamic
        glimpse_K = glimpse_K_static + glimpse_K_dynamic
        logit_K = logit_K_static + logit_K_dynamic
        query = fixed_context + projected_current_vehicle_state
        glimpse_Q = query.view(batch_size, num_vehicles, n_heads, 1, key_size)
        glimpse_Q = glimpse_Q.permute(2,0,1,3,4).contiguous()
        compatibility = glimpse_Q@glimpse_K.permute(0,1,2,4,3).contiguous() / math.sqrt(glimpse_Q.size(-1)) # glimpse_K => n_heads, batch_size, num_items, embed_dim
        compatibility = compatibility + feasibility_mask.unsqueeze(0).unsqueeze(3).float().log()
        
        attention = torch.softmax(compatibility.view(n_heads,batch_size,num_vehicles*num_nodes),dim=-1).view_as(compatibility)
        heads = attention@glimpse_V
        concated_heads = heads.permute(1,2,3,0,4).contiguous()
        concated_heads = concated_heads.view(batch_size, num_vehicles, 1, self.embed_dim)
        if param_dict is not None:
            final_Q = F.linear(concated_heads, param_dict["po_weight"])
        else:
            final_Q = self.project_out(concated_heads)
        logits = final_Q@logit_K.permute(0,1,3,2) / math.sqrt(final_Q.size(-1)) #batch_size, num_items, embed_dim
        logits = torch.tanh(logits) * self.tanh_clip
        logits = logits.squeeze(2) + feasibility_mask.float().log()
        logits = logits.view(batch_size, num_vehicles*num_nodes)
        probs = torch.softmax(logits, dim=1)
        return probs
    # @torch.jit.ignore
    def select(self, probs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        ### Select next to be executed.
        -----
        Parameter:
            probs: probabilities of each operation

        Return: index of operations, log of probabilities
        '''
        batch_size, num_choice = probs.shape
        batch_idx = torch.arange(batch_size, device=self.device)
        
        if self.training:
            dist = Categorical(probs.shape)
            dist.set_probs(probs)
            # dist = torch.distributions.Categorical(probs)
            op = dist.sample()
            while torch.any(probs[batch_idx, op[:]]==0):
                op = dist.sample()
            logprob = dist.log_prob(op)
            entropy = dist.entropy()
        else:
            prob, op = torch.max(probs, dim=-1)
            logprob = torch.log(prob)
            entropy = -torch.sum(prob*logprob)
        return op, logprob, entropy