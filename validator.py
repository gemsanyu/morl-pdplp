import os
import pathlib

import numpy as np
import torch
from pymoo.indicators.igd import IGD

from policy.non_dominated_sorting import fast_non_dominated_sort

"""
omega is the patience length for
early stopping
"""
class Validator:
    def __init__(self,
                 omega:int,
                 num_validation_samples:int,
                 num_objectives:int=2):
        self.omega = omega
        self.num_validation_samples = num_validation_samples
        self.num_objectives = num_objectives
        self.nadir_points = None
        self.utopia_points = None
        self.delta_nadir = None
        self.delta_utopia = None
        self.mean_delta_nadir = None
        self.mean_delta_utopia = None
        self.nd_solutions_list = None
        self.running_igd = None
        self.mean_running_igd = None
        self.epoch = 0
    
    @property
    def is_improving(self):
        if self.mean_running_igd is None or len(self.mean_running_igd) < self.omega:
            return True
        max_running_igd = float(np.max(self.mean_running_igd))
        max_delta_nadir = float(np.max(self.mean_delta_nadir))
        max_delta_utopia = float(np.max(self.mean_delta_utopia))
        is_convergence_done = (max_delta_nadir<=0.05) and (max_delta_utopia<=0.05)
        is_diversification_done = max_running_igd<=0.05
        return not (is_convergence_done and is_diversification_done)
    
    def get_last_mean_running_igd(self):
        if self.mean_running_igd is None:
            return None
        return self.mean_running_igd[-1]

    def get_last_delta_refpoints(self):
        if self.mean_delta_nadir is None:
            return None, None
        return self.mean_delta_nadir[-1], self.mean_delta_utopia[-1]

    def insert_new_nd_solutions(self, nd_solutions_list):
        if self.nd_solutions_list is None:
            self.nd_solutions_list = [nd_solutions_list]
        else:
            self.nd_solutions_list += [nd_solutions_list]
        if len(self.nd_solutions_list) > self.omega:
            self.nd_solutions_list = self.nd_solutions_list[-self.omega:]
        self.update_running_igd()
        

    def insert_new_ref_points(self, new_nadir_points, new_utopia_points):
        new_nadir_points = new_nadir_points.copy()
        new_utopia_points = new_utopia_points.copy()
        if self.nadir_points is None:
            self.nadir_points = new_nadir_points[np.newaxis, :, :]
            self.utopia_points = new_utopia_points[np.newaxis, :, :]
        else:
            last_nadir_points = self.nadir_points[-1]
            last_utopia_points = self.utopia_points[-1]
            new_nadir_points = np.maximum(last_nadir_points, new_nadir_points)
            new_utopia_points = np.minimum(new_utopia_points, last_utopia_points)
            self.nadir_points = np.concatenate([self.nadir_points, new_nadir_points[np.newaxis, :, :]], axis=0)
            self.utopia_points = np.concatenate([self.utopia_points, new_utopia_points[np.newaxis, :, :]], axis=0)
        if len(self.nadir_points) > self.omega:
            self.nadir_points = self.nadir_points[-self.omega:]
            self.utopia_points = self.utopia_points[-self.omega:]
        self.update_delta_ref_points()

    def update_running_igd(self):
        if len(self.nd_solutions_list) == 1 or self.nd_solutions_list is None:
            return
        num_problems = len(self.nd_solutions_list[0])
        running_igd_list = []
        for i in range(num_problems):
            current_nd_solutions = self.nd_solutions_list[-1][i]
            last_nd_solutions = self.nd_solutions_list[-2][i]
            nadir_points = self.nadir_points[-1,i,np.newaxis,:]
            utopia_points = self.utopia_points[-1,i,np.newaxis,:]
            denom = nadir_points-utopia_points
            current_nd_solutions = (current_nd_solutions-utopia_points)/denom
            last_nd_solutions = (last_nd_solutions-utopia_points)/denom
            combined_nd = np.concatenate([current_nd_solutions, last_nd_solutions], axis=0)
            nondom_idx = fast_non_dominated_sort(combined_nd)[0]
            combined_nd = combined_nd[nondom_idx]
            igd_getter = IGD(pf=combined_nd)
            running_igd = igd_getter._do(last_nd_solutions)
            running_igd_list += [running_igd]
        running_igd_list = np.asanyarray([running_igd_list])
        if self.running_igd is None:
            self.running_igd = running_igd_list
        else:
            self.running_igd = np.concatenate([self.running_igd, running_igd_list], axis=0)
        new_mean_running_igd = np.mean(self.running_igd[-1], keepdims=True)
        if self.mean_running_igd is None:
            self.mean_running_igd = new_mean_running_igd
        else:
            self.mean_running_igd = np.concatenate([self.mean_running_igd, new_mean_running_igd],axis=0)
        if len(self.running_igd) > self.omega:
            self.running_igd = self.running_igd[-self.omega:]
            self.mean_running_igd = self.mean_running_igd[-self.omega:]
        
    def update_delta_ref_points(self):
        if len(self.nadir_points) == 1 or self.nadir_points is None:
            return
        len_history = len(self.nadir_points)
        numerator_nadir = np.abs(self.nadir_points[len_history-2] - self.nadir_points[len_history-1])
        numerator_utopia = np.abs(self.utopia_points[len_history-2] - self.utopia_points[len_history-1])
        denominator = self.nadir_points[len_history-1] - self.utopia_points[len_history-1]
        denominator[denominator==0] = 1
        new_delta_nadir = np.max(numerator_nadir/denominator, axis=-1)[np.newaxis, :]
        new_delta_utopia = np.max(numerator_utopia/denominator, axis=-1)[np.newaxis, :]
        if self.delta_nadir is None:
            self.delta_nadir = new_delta_nadir
            self.delta_utopia = new_delta_utopia
            self.mean_delta_nadir = np.mean(new_delta_nadir)[np.newaxis]
            self.mean_delta_utopia = np.mean(new_delta_utopia)[np.newaxis]
        else:
            self.delta_nadir = np.concatenate([self.delta_nadir, new_delta_nadir], axis=0)
            self.delta_utopia = np.concatenate([self.delta_utopia, new_delta_utopia], axis=0)        
            self.mean_delta_nadir =  np.concatenate([self.mean_delta_nadir, np.mean(new_delta_nadir)[np.newaxis]])
            self.mean_delta_utopia = np.concatenate([self.mean_delta_utopia, np.mean(new_delta_utopia)[np.newaxis]])
        if len(self.delta_nadir) > self.omega:
            self.delta_nadir = self.delta_nadir[-self.omega:]
            self.delta_utopia = self.delta_utopia[-self.omega:]
            self.mean_delta_nadir = self.mean_delta_nadir[-self.omega:]
            self.mean_delta_utopia = self.mean_delta_utopia[-self.omega:]

def load_validator(args) -> Validator:
    title = args.title
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validator_path = checkpoint_dir/(title+"_validator.pt")
    validator = Validator(args.omega, args.num_validation_samples)
    if os.path.isfile(validator_path.absolute()):
        validator = torch.load(validator_path.absolute())
    return validator

def save_validator(validator, title):
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validator_path = checkpoint_dir/(title+"_validator.pt")
    validator_path_ = checkpoint_dir/(title+"_validator.pt_") #backup
    torch.save(validator, validator_path.absolute())
    torch.save(validator, validator_path_.absolute())