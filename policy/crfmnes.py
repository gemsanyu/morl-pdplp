import math

import numpy as np
import torch

from model.agent import Agent
from policy.policy import Policy


# evaluation value of the infeasible solution
INFEASIBLE = np.inf

def get_h_inv(dim):
    f = lambda a, b: ((1. + a * a) * math.exp(a * a / 2.) / 0.24) - 10. - dim
    f_prime = lambda a: (1. / 0.24) * a * math.exp(a * a / 2.) * (3. + a * a)
    h_inv = 6.0
    while abs(f(h_inv, dim)) > 1e-10:
        last = h_inv
        h_inv = h_inv - 0.5 * (f(h_inv, dim) / f_prime(h_inv))
        if abs(h_inv - last) < 1e-16:
            # Exit early since no further improvements are happening
            break
    return h_inv

class CRFMNES(Policy):
    def __init__(self, 
                num_neurons, 
                num_vehicle_dynamic_features, 
                num_node_dynamic_features,
                sigma,
                negative_hv):
        super().__init__(num_neurons, num_vehicle_dynamic_features, num_node_dynamic_features)
        self.sigma = sigma
        self.negative_hv = negative_hv
        self.v = np.random.randn(self.n_params, 1) / np.sqrt(self.n_params)
        self.D = np.ones([self.n_params, 1])
        self.m = np.zeros([self.n_params, 1])
        # self.w_rank_hat = (np.log(self.lamb / 2 + 1) - np.log(np.arange(1, self.lamb + 1))).reshape(self.lamb, 1)
        # self.w_rank_hat[np.where(self.w_rank_hat < 0)] = 0
        # self.w_rank = self.w_rank_hat / sum(self.w_rank_hat) - (1. / self.lamb)
        # self.mueff = 1 / ((self.w_rank + (1 / self.lamb)).T @ (self.w_rank + (1 / self.lamb)))[0][0]
        # self.cs = (self.mueff + 2.) / (self.n_params + self.mueff + 5.)
        # self.cc = (4. + self.mueff / self.n_params) / (self.n_params + 4. + 2. * self.mueff / self.n_params)
        # self.c1_cma = 2. / (math.pow(self.n_params + 1.3, 2) + self.mueff)
        # initialization
        self.chiN = np.sqrt(self.n_params) * (1. - 1. / (4. * self.n_params) + 1. / (21. * self.n_params * self.n_params))
        self.pc = np.zeros([self.n_params, 1])
        self.ps = np.zeros([self.n_params, 1])
        # distance weight parameter
        self.h_inv = get_h_inv(self.n_params)
        # self.alpha_dist = lambda lambF: self.h_inv * min(1., math.sqrt(float(self.lamb) / self.n_params)) * math.sqrt(
        #     float(lambF) / self.lamb)
        # self.w_dist_hat = lambda z, lambF: math.exp(self.alpha_dist(lambF) * np.linalg.norm(z))
        # learning rate
        self.eta_m = 1.0
        self.eta_move_sigma = 1.
        self.eta_stag_sigma = lambda lambF: math.tanh((0.024 * lambF + 0.7 * self.n_params + 20.) / (self.n_params + 12.))
        self.eta_conv_sigma = lambda lambF: 2. * math.tanh((0.025 * lambF + 0.75 * self.n_params + 10.) / (self.n_params + 4.))
        # self.c1 = lambda lambF: self.c1_cma * (self.n_params - 5) / 6 * (float(lambF) / self.lamb)
        self.eta_B = lambda lambF: np.tanh((min(0.02 * lambF, 3 * np.log(self.n_params)) + 5) / (0.23 * self.n_params + 25))

        self.g = 0
        self.no_of_evals = 0

        # self.idxp = np.arange(self.lamb / 2, dtype=int)
        # self.idxm = np.arange(self.lamb / 2, self.lamb, dtype=int)
        # self.z = np.zeros([self.n_params, self.lamb])
        
    def copy_to_mean(self, agent: Agent):
        pcs_weight, pns_weight, po_weight = None, None, None
        for name, param in agent.named_parameters():
            if name == "project_current_vehicle_state.weight":
                pcs_weight = param.data.ravel()
            if name == "project_node_state.weight":
                pns_weight = param.data.ravel()
            if name == "project_out.weight":
                po_weight = param.data.ravel()
        mu_list = []
        mu_list += [pcs_weight]
        mu_list += [pns_weight]
        mu_list += [po_weight]
        m = torch.cat(mu_list)
        m = m.unsqueeze(0).numpy()
        self.m = m.reshape(self.n_params, 1)

    def generate_random_parameters(self, n_sample: int = 2, use_antithetic=True):
        zhalf = np.random.randn(self.n_params, int(n_sample / 2))  # dim x lamb/2
        z = np.concatenate([zhalf, -zhalf], axis=-1)
        normv = np.linalg.norm(self.v)
        normv2 = normv ** 2
        vbar = self.v / normv
        y = z + (np.sqrt(1 + normv2) - 1) * vbar @ (vbar.T @ z)
        x = self.m + self.sigma * y * self.D
        param_dict_list = [self.create_param_dict(torch.from_numpy(x[:, i]).to(torch.float32)) for i in range(n_sample)]
        return param_dict_list, (x,y,z)

    def update(self, random_params, score):
        score = score.ravel()
        x,y,z = random_params
        normv = np.linalg.norm(self.v)
        normv2 = normv ** 2
        vbar = self.v / normv
        _, n_sample = x.shape
        sorted_indices = np.argsort(score)
        z = z[:, sorted_indices]
        y = y[:, sorted_indices]
        x = x[:, sorted_indices]

        # calculate what
        self.w_rank_hat = (np.log(n_sample / 2 + 1) - np.log(np.arange(1, n_sample + 1))).reshape(n_sample, 1)
        self.w_rank_hat[np.where(self.w_rank_hat < 0)] = 0
        self.w_rank = self.w_rank_hat / sum(self.w_rank_hat) - (1. / n_sample)
        self.mueff = 1 / ((self.w_rank + (1 / n_sample)).T @ (self.w_rank + (1 / n_sample)))[0][0]
        self.cs = (self.mueff + 2.) / (self.n_params + self.mueff + 5.)
        self.cc = (4. + self.mueff / self.n_params) / (self.n_params + 4. + 2. * self.mueff / self.n_params)
        self.c1_cma = 2. / (math.pow(self.n_params + 1.3, 2) + self.mueff)
        self.alpha_dist = lambda lambF: self.h_inv * min(1., math.sqrt(float(n_sample) / self.n_params)) * math.sqrt(
            float(lambF) / n_sample)
        self.w_dist_hat = lambda z, lambF: math.exp(self.alpha_dist(lambF) * np.linalg.norm(z))
        self.c1 = lambda lambF: self.c1_cma * (self.n_params - 5) / 6 * (float(lambF) / n_sample)
        
        # This operation assumes that if the solution is infeasible, infinity comes in as input.
        lambF = n_sample

        # evolution path p_sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2. - self.cs) * self.mueff) * (z @ self.w_rank)
        ps_norm = np.linalg.norm(self.ps)
        # distance weight
        w_tmp = np.array(
            [self.w_rank_hat[i] * self.w_dist_hat(np.array(z[:, i]), lambF) for i in range(n_sample)]).reshape(
            n_sample, 1)
        weights_dist = w_tmp / sum(w_tmp) - 1. / n_sample
        # switching weights and learning rate
        weights = weights_dist if ps_norm >= self.chiN else self.w_rank
        eta_sigma = self.eta_move_sigma if ps_norm >= self.chiN else self.eta_stag_sigma(
            lambF) if ps_norm >= 0.1 * self.chiN else self.eta_conv_sigma(lambF)
        # update pc, m
        wxm = (x - self.m) @ weights
        self.pc = (1. - self.cc) * self.pc + np.sqrt(self.cc * (2. - self.cc) * self.mueff) * wxm / self.sigma
        self.m += self.eta_m * wxm
        # calculate s, t
        # step1
        normv4 = normv2 ** 2
        exY = np.append(y, self.pc / self.D, axis=1)  # dim x lamb+1
        yy = exY * exY  # dim x lamb+1
        ip_yvbar = vbar.T @ exY
        yvbar = exY * vbar  # dim x lamb+1. exYのそれぞれの列にvbarがかかる
        gammav = 1. + normv2
        vbarbar = vbar * vbar
        alphavd = np.min(
            [1, np.sqrt(normv4 + (2 * gammav - np.sqrt(gammav)) / np.max(vbarbar)) / (2 + normv2)])  # scalar
        t = exY * ip_yvbar - vbar * (ip_yvbar ** 2 + gammav) / 2  # dim x lamb+1
        b = -(1 - alphavd ** 2) * normv4 / gammav + 2 * alphavd ** 2
        H = np.ones([self.n_params, 1]) * 2 - (b + 2 * alphavd ** 2) * vbarbar  # dim x 1
        invH = H ** (-1)
        s_step1 = yy - normv2 / gammav * (yvbar * ip_yvbar) - np.ones([self.n_params, n_sample + 1])  # dim x lamb+1
        ip_vbart = vbar.T @ t  # 1 x lamb+1
        s_step2 = s_step1 - alphavd / gammav * ((2 + normv2) * (t * vbar) - normv2 * vbarbar @ ip_vbart)  # dim x lamb+1
        invHvbarbar = invH * vbarbar
        ip_s_step2invHvbarbar = invHvbarbar.T @ s_step2  # 1 x lamb+1
        s = (s_step2 * invH) - b / (
                    1 + b * vbarbar.T @ invHvbarbar) * invHvbarbar @ ip_s_step2invHvbarbar  # dim x lamb+1
        ip_svbarbar = vbarbar.T @ s  # 1 x lamb+1
        t = t - alphavd * ((2 + normv2) * (s * vbar) - vbar @ ip_svbarbar)  # dim x lamb+1
        # update v, D
        exw = np.append(self.eta_B(lambF) * weights, np.array([self.c1(lambF)]).reshape(1, 1),
                        axis=0)  # lamb+1 x 1
        self.v = self.v + (t @ exw) / normv
        self.D = self.D + (s @ exw) * self.D
        # calculate detA
        nthrootdetA = np.exp(np.sum(np.log(self.D)) / self.n_params + np.log(1 + self.v.T @ self.v) / (2 * self.n_params))[0][0]
        self.D = self.D / nthrootdetA
        # update sigma
        G_s = np.sum((z * z - np.ones([self.n_params, n_sample])) @ weights) / self.n_params
        self.sigma = self.sigma * np.exp(eta_sigma / 2 * G_s)