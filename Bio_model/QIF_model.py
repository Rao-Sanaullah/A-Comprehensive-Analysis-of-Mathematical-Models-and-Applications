import numpy as np

class QIF:
    def __init__(self, tau, v_reset, v_th, v_init=0.0, beta=0.5, n_neurons=50):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = np.full(n_neurons, v_init)
        self.beta = beta

    def update(self, I, dt):
        dvdt = (-self.v + self.beta * self.v**2 + I) / self.tau
        self.v += dvdt * dt
        spikes = self.v >= self.v_th
        self.v = np.where(spikes, self.v_reset, self.v)
        return spikes
