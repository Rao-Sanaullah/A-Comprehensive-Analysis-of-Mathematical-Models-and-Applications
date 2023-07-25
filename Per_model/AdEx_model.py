import numpy as np

# define AdEx model

class AdEX:
    def __init__(self, tau_m, v_rheo, v_spike, delta_T, v_reset, v_init=0.0, n_neurons=50):
        self.tau_m = tau_m
        self.v_rheo = v_rheo
        self.v_spike = v_spike
        self.delta_T = delta_T
        self.v_reset = v_reset
        self.v = v_init
        self.n_neurons = n_neurons
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))
        
    def update(self, I, dt):
        dvdt = (-self.v + self.tau_m * I - self.v_rheo + self.delta_T * np.exp((self.v - self.v_spike) / self.delta_T)) / self.tau_m
        self.v += dvdt * dt
        spike = self.v >= self.v_spike
        self.v = np.where(spike, self.v_reset, self.v)
        return spike