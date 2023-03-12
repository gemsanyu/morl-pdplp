from abc import abstractmethod
import math
from typing import List

import torch


CPU_DEVICE = torch.device("cpu")
# ES object
# generate parameters, map parameters, replace paramater of a model
# update parameters


class Policy(object):
    def __init__(self, num_neurons, num_vehicle_dynamic_features, num_node_dynamic_features):
        self.num_neurons = num_neurons
        self.num_node_dynamic_features = num_node_dynamic_features
        self.num_vehicle_dynamic_features = num_vehicle_dynamic_features
        self.vehicle_current_state_dim = num_neurons + num_vehicle_dynamic_features 
        # self.n_params = 3a^2+a*csd+3a*ndf+a^2 = 4a^2+a*(csd+3ndf)  

        #hitung per component agar mudah plug in plug out component uji coba
        self.n_params = 0
        self.n_params += self.num_neurons*self.vehicle_current_state_dim
        self.n_params += 3*self.num_neurons*self.num_node_dynamic_features
        self.n_params += self.num_neurons*self.num_neurons

    def create_param_dict(self, param_vec):
        param_vec = param_vec.ravel()
        params_idx = 0
        pcs_weight = param_vec[params_idx:params_idx+self.num_neurons*self.vehicle_current_state_dim].view(self.num_neurons,self.vehicle_current_state_dim)
        params_idx += self.num_neurons*self.vehicle_current_state_dim
        pns_weight = param_vec[params_idx:params_idx+3*self.num_neurons*self.num_node_dynamic_features].view(3*self.num_neurons, self.num_node_dynamic_features)
        params_idx += 3*self.num_neurons*self.num_node_dynamic_features
        po_weight = param_vec[params_idx:params_idx+self.num_neurons*self.num_neurons].view(self.num_neurons, self.num_neurons)
        params_idx += self.num_neurons*self.num_neurons
        param_dict = {
                     "pcs_weight":pcs_weight,
                     "pns_weight":pns_weight,
                     "po_weight":po_weight,
                     }      
        return param_dict

    @abstractmethod
    def generate_random_parameters(self, n_sample):
        pass

    @abstractmethod
    def generate_on_mean(self):
        pass

    @abstractmethod
    def logprob(self, sample_list):
        pass

    @abstractmethod
    def set_new_device(self, new_device, actor_device=None):
        pass


def get_multi_importance_weight(policy_list, sample_list):
    num_sample, _ = sample_list.shape
    num_policy = len(policy_list)
    logprobs = torch.zeros((num_policy, num_sample),
                           dtype=torch.float32)
    for i in range(len(policy_list)):
        logprobs[i, :] = policy_list[i].logprob(sample_list)

    weights = torch.softmax(logprobs, dim=0)
    # get the first weights, because the first weights is the current policy
    weights = weights[0, :]
    weights *= num_policy
    return weights.unsqueeze(1)
