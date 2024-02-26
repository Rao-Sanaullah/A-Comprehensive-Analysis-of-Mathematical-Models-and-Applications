import numpy as np

class AdEX:
    def __init__(self, tau_m, v_rheo, v_spike, delta_T, v_reset, v_init=0.0, n_neurons=1000):
        self.tau_m = tau_m
        self.v_rheo = v_rheo
        self.v_spike = v_spike
        self.delta_T = delta_T
        self.v_reset = v_reset
        self.v = np.full(n_neurons, v_init)  # Array of initial membrane potentials
        self.n_neurons = n_neurons

    def update(self, I, dt):
        dvdt = (-self.v + self.tau_m * I - self.v_rheo + self.delta_T * np.exp((self.v - self.v_spike) / self.delta_T)) / self.tau_m
        self.v += dvdt * dt
        spikes = self.v >= self.v_spike
        self.v = np.where(spikes, self.v_reset, self.v)  # Reset membrane potential for spiking neurons
        return spikes
