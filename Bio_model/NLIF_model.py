import numpy as np

# define NLIF model
class NLIF:
    def __init__(self, tau, v_reset, v_th, v_init=0.0, alpha=1.0, beta=1.0, n_neurons=50):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.alpha = alpha
        self.beta = beta
        self.n_neurons = n_neurons
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))

    def update(self, I, dt):
        dvdt = (-self.v + I) / self.tau
        self.v += dvdt * dt
        spike = self.v >= self.v_th
        self.v = np.where(spike, self.v_reset + self.alpha * (self.v - self.v_th), self.v * self.beta)
        return spike