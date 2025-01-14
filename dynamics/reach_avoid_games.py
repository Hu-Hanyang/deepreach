import torch
from dynamics.dynamics import Dynamics

"""
Reach-Avoid differential game. 
Naming convention: number dynamcis vs number dynamics

"""
class OneSIGvsOneSIG(Dynamics):
    def __init__(self, speed_a:float = 1.0, speed_d: float = 1.5, 
                 u_min: float = -1.0, u_max: float = 1.0, 
                 d_min: float = -1.0, d_max: float = 1.0):
        self.va = speed_a  # Attacker speed
        self.vd = speed_d  # Defender speed
        self.u_range = [u_min, u_max]
        self.d_range = [d_min, d_max]
        
        super().__init__(
            loss_type='brat_hjivi', set_mode='reach',
            state_dim=4, input_dim=5, control_dim=2, disturbance_dim=2,
            state_mean=[0, 0, 0, 0],
            state_var=[1, 1, 1, 1],  # x,y in [-1, +1]
            value_mean=0.0,  #TODO: not sure about definition
            value_var=1.0,  #TODO: not sure about definition
            value_normto=0.02, #TODO: not sure about definition 
            deepreach_model="exact"
        )
        
    def state_test_range(self):
        return [
            [-1, +1],
            [-1, +1],
            [-1, +1],
        ]
        
    def equivalent_wrapped_state(self, state):
        return state
    
    # \dot x_a = v_a * u1
    # \dot y_a = v_a * u2
    # \dot x_d = v_d * d1
    # \dot y_d = v_d * d2
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.va * control[..., 0]  # dx_a
        dsdt[..., 1] = self.va * control[..., 1]  # dy_a
        dsdt[..., 2] = self.vd * disturbance[..., 0]  # dx_d
        dsdt[..., 3] = self.vd * disturbance[..., 1]  # dy_d
        return dsdt
    
    def reach_fn(self, state):
        #TODO: need to check, not sure
        # # goal1: circle
        # center = torch.tensor([0.7, 0.2])
        # goalR = 0.1
        # return torch.norm(state[..., :2] - center, dim=-1) - goalR
        # goal2: rectangle
        upper_x = state[..., 0] - 0.8
        lower_x = 0.6 - state[..., 0]
        dist_x = torch.max(lower_x, upper_x)
        upper_y = state[..., 1] - 0.3
        lower_y = 0.1 - state[..., 1]
        dist_y = torch.max(lower_y, upper_y)
        max_dist = torch.max(dist_x, dist_y)

        return torch.where((max_dist<=0), max_dist/1.0, max_dist/1.0)
    
    def avoid_fn(self, state):
        # capture distance
        capture_dist = torch.norm(state[...,:2]-state[...,2:], dim=-1) - 0.1

        return capture_dist
    
    def boundary_fn(self, state):
        return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))
    
    def sample_target_state(self, num_samples):
        target_state_range = self.state_test_range()
        target_state_range[0] = [0.6, 0.8] # x in [0.6, 0.8]
        target_state_range[1] = [0.1, 0.3]  # y in [0.1, 0.3]
        target_state_range = torch.tensor(target_state_range)
        
        return target_state_range[:, 0] + torch.rand(num_samples, self.state_dim)*(target_state_range[:, 1] - target_state_range[:, 0])

    def cost_fn(self, state_traj):
        #TODO: what does this do?
        # return min_t max{l(x(t)), max_k_up_to_t{-g(x(k))}}, where l(x) is reach_fn, g(x) is avoid_fn 
        reach_values = self.reach_fn(state_traj)
        avoid_values = self.avoid_fn(state_traj)
        return torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values

    def hamiltonian(self, state, dvds):
        opt_ctrl = self.optimal_control(state, dvds)
        opt_distb = self.optimal_disturbance(state, dvds)

        u1_coeff = dvds[..., 0] * self.va * opt_ctrl[..., 0]
        u2_coeff = dvds[..., 1] * self.va * opt_ctrl[..., 1]
        d1_coeff = dvds[..., 2] * self.vd * opt_distb[..., 0]
        d2_coeff = dvds[..., 3] * self.vd * opt_distb[..., 1]

        return u1_coeff + u2_coeff + d1_coeff + d2_coeff
    
    def optimal_control(self, state, dvds):
        # control aims to minimize the value
        ctrl_len = torch.norm(dvds[..., 0] * dvds[..., 0] + dvds[..., 1] * dvds[..., 1], dim=-1)
        opt_ctrl = self.u_range[0] * torch.abs(dvds[..., :2]) / ctrl_len

        return opt_ctrl
    
    def optimal_disturbance(self, state, dvds):
        # disturbance aims to maximize the value
        distb_len = torch.norm(dvds[..., 2] * dvds[..., 2] + dvds[..., 3] * dvds[..., 3], dim=-1)
        opt_distb = self.d_range[1] * torch.abs(dvds[..., 2:]) / distb_len
        
        return opt_distb
    
    def plot_config(self):
        return {
            'state_slices': [-0.5, 0, 0.5, -0.5],
            'state_labels': ['xa', 'ya', 'xd', 'yd'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }
        