import torch
from dynamics.dynamics import Dynamics

#TODO add disturbances to the dynamics modules in dynamics.py
from dynamics.dynamics import Dubins3D


class Dubins3DDistb(Dubins3D):
    def __init__(self, goalR: float, velocity: float, omega_max: float, angle_alpha_factor: float, set_mode: str, freeze_model: bool, disturbance_dim:int, d_min:float, d_max:float):
        super().__init__(goalR, velocity, omega_max, angle_alpha_factor, set_mode, freeze_model)
        self.d_range = [d_min, d_max]
        self.disturbance_dim = disturbance_dim  # override the Dynamics class
    # Dubins3D dynamics with disturbances
    # \dot x    = v \cos \theta + d1
    # \dot y    = v \sin \theta + d2
    # \dot \theta = u 
    def dsdt(self, state, control, disturbance):
        if self.freeze_model:
            raise NotImplementedError
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 2]) + disturbance[0]
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 2]) + disturbance[1]
        dsdt[..., 2] = control[..., 0]
        
        return dsdt

    def optimal_disturbance(self, state, dvds):
        #TODO: how to tackle with the control input?
        if self.set_mode == 'reach':  # if control aims to minimize, the distb aims to maximize
            return
        elif self.set_mode == 'avoid':
            
            return 