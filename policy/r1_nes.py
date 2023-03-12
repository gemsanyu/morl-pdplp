import math

import torch
import matplotlib.pyplot as plt

from model.agent import Agent
from policy.policy import Policy

CPU_DEVICE = torch.device("cpu")
# ES object
# generate parameters, map parameters, replace paramater of a model
# update parameters


class R1_NES(Policy):
    def __init__(self,
                 num_neurons,
                 num_vehicle_dynamic_features,
                 num_node_dynamic_features,
                 ld,
                 negative_hv,
                 lr=None,
                 pop_size=None
                 ):
        super(R1_NES, self).__init__(num_neurons, num_vehicle_dynamic_features, num_node_dynamic_features)

        stdv  = 1./math.sqrt(self.n_params)
        self.mu = torch.rand(size=(1, self.n_params), dtype=torch.float32)*2*stdv-stdv
        self.ld = ld
        self.principal_vector = torch.randn(size=(self.n_params,), dtype=torch.float32)
        self.principal_vector /= torch.norm(self.principal_vector)

        # hyperparams
        self.negative_hv = negative_hv
        self.lr_mu = 1
        # old self.lr = (3+math.log(self.n_params))/(5*math.sqrt(self.n_params))
                
        if pop_size is None:
            pop_size = int(4*math.log2(self.n_params))
        self.pop_size = pop_size
        if lr is None:
            lr = 0.6 * (3 + math.log2(self.n_params)) / self.n_params / math.sqrt(self.n_params)
        self.lr = lr

    def copy_to_mu(self, agent: Agent):
        for name, param in agent.named_parameters():
            if name == "project_current_state.weight":
                pcs_weight = param.data.ravel()
            if name == "project_node_state.weight":
                pns_weight = param.data.ravel()
            if name == "project_out.weight":
                po_weight = param.data.ravel()
        mu_list = []
        mu_list += [pcs_weight]
        mu_list += [pns_weight]
        mu_list += [po_weight]
        self.mu = torch.cat(mu_list)
        self.mu = self.mu.unsqueeze(0)


    '''
    y ~ N(0,I)
    k ~ N(0,1)
    theta = mu + s*sigma
    return theta mapped with param names of the policy
    '''
    def generate_random_parameters(self, n_sample: int = 2, use_antithetic=True):
        
        if n_sample > 1:
            if use_antithetic:
                y_list = self.norm_dist.sample(
                    (int(n_sample/2), self.n_params))
                y_list = torch.cat((y_list, -y_list), dim=0)
                k_list = self.norm_dist.sample((int(n_sample/2), 1))
                k_list = torch.cat((k_list, -k_list), dim=0)
            else:
                y_list = [torch.randn(1, self.n_params) for _ in range(n_sample)]
                k_list = [torch.randn(1, self.n_params) for _ in range(n_sample)]
                y_list = torch.cat(y_list, dim=0)
                k_list = torch.cat(k_list, dim=0)
                # y_list = self.norm_dist.sample((n_sample, self.n_params))
                # k_list = self.norm_dist.sample((n_sample, 1))
        else:
            y_list = self.norm_dist.sample((1, self.n_params))
            k_list = self.norm_dist.sample((1, 1))
        g = math.exp(self.ld) * (y_list + k_list*self.principal_vector)
        random_params = self.mu + g

        param_dict_list = []
        for param_vec in random_params:
            param_dict_list += [self.create_param_dict(param_vec)]
        return param_dict_list, random_params

    '''
    theta = mu
    return theta mapped with param names of the policy
    '''
    def generate_on_mean(self):
        param_dict = self.create_param_dict(self.mu)
        return param_dict

    # update given the values
    def update(self, w_list, x_list, score):
        score = torch.from_numpy(score)
        # prepare natural gradients
        d = self.n_params
        r = torch.norm(self.principal_vector)
        u = self.principal_vector/r
        wtw = torch.sum(w_list*w_list, dim=1, keepdim=True)
        wtu = torch.sum(w_list*u, dim=1, keepdim=True)
        wtu2 = wtu**2

        ngrad_mu_l = x_list
        ngrad_ld_l = 1/(2*(d-1)) * ((wtw-d) - (wtu2-1))
        ngrad_pv_l = ((r**2-d+2)*wtu2-(r**2+1)*wtw) * \
            u/(2*r*(d-1)) + (wtu*w_list)/r
        ngrad_pv_j = torch.sum(score*ngrad_pv_l, dim=0, keepdim=True)
        nvtz = torch.sum(ngrad_pv_l*u, dim=1, keepdim=True)
        ngrad_c_l = nvtz/r
        ngrad_z_l = (ngrad_pv_l - nvtz*u)/r
        ngrad_c_j = torch.sum(score*ngrad_c_l, dim=0)
        ngrad_z_j = torch.sum(score*ngrad_z_l, dim=0, keepdim=True)
        # print(ngrad_c_j)
        # print(ngrad_z_j)
        print(self.principal_vector)
        # start updating
        # conditional update on c,z,v to prevent unstable (flipping and large) v update
        epsilon = min(self.lr, 2 * math.sqrt(r ** 2 / torch.sum(ngrad_pv_j**2)))
        if ngrad_c_j < 0:
            # multiplicative update
            c = torch.log(r)
            c = c + epsilon*ngrad_c_j
            z = u + epsilon*ngrad_z_j
            z = z/torch.norm(z)
            self.principal_vector = torch.exp(c)*z
        else:
            # additive update
            self.principal_vector = self.principal_vector + epsilon*ngrad_pv_j
        print(self.principal_vector)
        ngrad_mu_j = torch.sum(score*ngrad_mu_l, dim=0)
        ngrad_ld_j = torch.sum(score*ngrad_ld_l, dim=0)
        print(self.mu)
        self.mu = self.mu + self.lr_mu*ngrad_mu_j
        print(self.mu)
        print(self.ld)
        self.ld = self.ld + self.lr*ngrad_ld_j
        print(self.ld)
        exit()

    @property
    def _getMaxVariance(self):
        return math.exp(self.ld * 2 / self.n_params)

    def logprob(self, sample_list):
        x_list = sample_list
        r = torch.norm(self.principal_vector)
        xtx = torch.sum(x_list*x_list, dim=1)
        xtv = torch.sum(x_list*self.principal_vector, dim=1)

        cc = self.n_params*math.log(2*math.pi)
        temp1 = -self.ld*self.n_params - \
            torch.log(1+r**2)/2 - math.exp(-2*self.ld)*xtx/2
        temp2 = ((math.exp(-2*self.ld))/(2*(1+r**2)))*xtv**2
        logprob = cc + temp1 + temp2
        return logprob

    def write_progress_to_tb(self, writer, step):
        # note the parameters
        writer.add_scalar("Mu Norm", torch.norm(self.mu).cpu().item(), step)
        writer.add_scalar("V Norm", torch.norm(self.principal_vector).cpu().item(), step)
        writer.add_scalar("Max Var", self._getMaxVariance, step)    
        writer.add_scalar("Lambda", self.ld, step)
        maxmu = torch.max(self.mu)
        minmu = torch.min(self.mu)
        writer.add_scalar("Max Mu", maxmu.item(), step)
        writer.add_scalar("Min Mu", minmu.item(), step)
        writer.flush()

'''
Lets separate experience replay for each policy, so that we can save whatever needed
without preprocessing in every update step to improve runtime.
'''
class ExperienceReplay(object):
    def __init__(self, dim, num_obj=2, max_saved_policy=5, num_sample=10, device=CPU_DEVICE):
        super().__init__()
        self.dim = dim
        self.max_saved_policy = max_saved_policy
        self.num_sample = num_sample
        self.num_obj = num_obj

        self.num_saved_policy = 0
        self.policy_list = []

        self.w_list = torch.zeros(
            (self.max_saved_policy*num_sample, dim), dtype=torch.float32)
        self.x_list = torch.zeros(
            (self.max_saved_policy*num_sample, dim), dtype=torch.float32)
        self.f_list = torch.zeros(
            (self.max_saved_policy*num_sample, num_obj), dtype=torch.float32)
        self.node_order_list = None
        self.item_selection_list = None

    def clear(self):
        self.num_saved_policy = 0
        self.node_order_list = None
        self.item_selection_list = None


    def add(self, policy, sample_list, f_list, node_order_list, item_selection_list):
        self.num_saved_policy = min(
            self.num_saved_policy+1, self.max_saved_policy)

        # push to front, while rolling to remove overflowing samples
        self.policy_list = [policy] + \
            self.policy_list[0:self.max_saved_policy-1]

        x_list = sample_list - policy.mu
        self.x_list = self.x_list.roll(self.num_sample, dims=0)
        self.x_list[:self.num_sample, :] = x_list

        w_list = x_list/math.exp(policy.ld)
        self.w_list = self.w_list.roll(self.num_sample, dims=0)
        self.w_list[:self.num_sample, :] = w_list

        self.f_list = self.f_list.roll(self.num_sample, dims=0)
        self.f_list[:self.num_sample] = f_list

        if self.node_order_list is None:
            num_sample, num_nodes = node_order_list.shape 
            self.node_order_list = torch.zeros((self.max_saved_policy*self.num_sample, num_nodes), dtype=torch.long)
        self.node_order_list = self.node_order_list.roll(self.num_sample, dims=0)
        self.node_order_list[:self.num_sample, :] = node_order_list

        if self.item_selection_list is None:
            num_sample, num_items = item_selection_list.shape 
            self.item_selection_list = torch.zeros((self.max_saved_policy*self.num_sample, num_items), dtype=torch.bool)
        self.item_selection_list = self.item_selection_list.roll(self.num_sample, dims=0)
        self.item_selection_list[:self.num_sample, :] = item_selection_list
