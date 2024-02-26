import numpy as np

# define SRM model
class SRM:
    def __init__(self, tau_s, tau_r, v_reset, v_th, v_init=0.0, n_neurons=50):
        self.tau_s = tau_s
        self.tau_r = tau_r
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.n_neurons = n_neurons
        self.s = np.zeros(n_neurons)
        self.r = np.zeros(n_neurons)
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))
        
    def update(self, I, dt):
        dsdt = -self.s / self.tau_s + self.r
        drdt = -self.r / self.tau_r
        self.s += dsdt * dt
        self.r += drdt * dt
        dvdt = (-self.v + np.dot(self.weights.T, self.s) + I) / self.tau_s
        self.v += dvdt * dt
        if self.v >= self.v_th:
            spike = True
            self.v = self.v_reset
            self.s += 1.0
            self.r += 1.0
        else:
            spike = False
        return spike