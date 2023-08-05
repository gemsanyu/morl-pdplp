import math

from typing import Dict
import torch as T
import torch.nn as nn

CPU_DEVICE = T.device("cpu")

# class PHN(T.jit.ScriptModule):
class PHN(T.nn.Module):
    def __init__(
            self,
            ray_hidden_size: int=128,
            num_neurons: int=64,
            device=CPU_DEVICE,
        ) -> None:
        '''
        ### Embedder class.
        -----
        It uses MLP method.

        Parameter:
            input_size: size for input in int
            hidden_layer_sizes: size for layers in hidden layer
        '''
        super(PHN, self).__init__()
        self.ray_layer = nn.Sequential(
                                        nn.Linear(2, ray_hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(ray_hidden_size, ray_hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(ray_hidden_size, ray_hidden_size))
        # self.pe_layer = nn.Sequential(
        #     nn.Linear(ray_hidden_size, ray_hidden_size),
        #     nn.Linear(ray_hidden_size, 3*num_neurons*num_neurons)
        # )
        # self.pf_layer = nn.Sequential(
        #     nn.Linear(ray_hidden_size, ray_hidden_size),
        #     nn.Linear(ray_hidden_size, num_neurons*num_neurons)
        # ) 
        # self.pcs_layer = nn.Sequential(
        #     nn.Linear(ray_hidden_size, ray_hidden_size),
        #     nn.Linear(ray_hidden_size, self.current_state_dim*num_neurons)
        # )
        # self.pns_layer = nn.Sequential(
        #     nn.Linear(ray_hidden_size, ray_hidden_size),
        #     nn.Linear(ray_hidden_size, self.num_node_dynamic_features*3*num_neurons)
        # )
        self.po_layer = nn.Sequential(
            nn.Linear(ray_hidden_size, ray_hidden_size),
            nn.Linear(ray_hidden_size, num_neurons*num_neurons)
        )
        self.ray_hidden_size = ray_hidden_size
        self.num_neurons = num_neurons
        self.device = device
        self.to(device)
        self.init_parameters()
    
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    # @T.jit.script_method
    def forward(self, ray: T.Tensor) -> Dict[str, T.Tensor]:
        '''
        ### Calculate embedding.
        -----

        Parameter:
            input: weight preferences/ray

        Return: appropriate weights
        '''
        ray_features = self.ray_layer(ray)
        # pcs_weight = self.pcs_layer(ray_features).view(self.num_neurons,self.current_state_dim)
        # pns_weight = self.pns_layer(ray_features).view(3*self.num_neurons, self.num_node_dynamic_features)
        po_weight = self.po_layer(ray_features).view(self.num_neurons, self.num_neurons)
        param_dict = {
                    #  "pcs_weight":pcs_weight,
                    #  "pns_weight":pns_weight,
                     "po_weight":po_weight,
                     }
        return param_dict
