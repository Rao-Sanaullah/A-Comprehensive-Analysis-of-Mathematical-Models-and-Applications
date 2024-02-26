import numpy as np

class LIF:
    def __init__(self, tau, v_reset, v_th, v_init=0.0, n_neurons=50):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = np.full(n_neurons, v_init)

    def update(self, I, dt):
        dvdt = (-self.v + I) / self.tau
        self.v += dvdt * dt
        spikes = self.v >= self.v_th
        self.v = np.where(spikes, self.v_reset, self.v)
        return spikes
