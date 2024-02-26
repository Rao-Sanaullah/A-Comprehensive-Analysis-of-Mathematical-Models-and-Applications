import numpy as np

# define AdEx model
class AdEX:
    def __init__(self, tau_m, v_rheo, v_spike, delta_T, v_reset, v_init=0.0):
        self.tau_m = tau_m
        self.v_rheo = v_rheo
        self.v_spike = v_spike
        self.delta_T = delta_T
        self.v_reset = v_reset
        self.v = v_init
        self.num_ops = 5  # one multiplication, four additions
        
    def update(self, I, dt):
        dvdt = (-self.v + self.tau_m * I - self.v_rheo + self.delta_T * np.exp((self.v - self.v_spike) / self.delta_T)) / self.tau_m
        self.v += dvdt * dt
        spike = self.v >= self.v_spike
        if spike:
            self.v = self.v_reset
            self.num_ops += 4  # two multiplications, two additions
        else:
            self.num_ops += 1  # one multiplication
        self.num_ops += 3  # three additions
        return spike
