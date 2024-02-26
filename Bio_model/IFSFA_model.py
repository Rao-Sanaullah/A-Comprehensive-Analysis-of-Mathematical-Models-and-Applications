import numpy as np

class IFSFA:
    def __init__(self, tau_m, tau_w, a, b, delta_T, v_reset, v_th, v_init=0.0, n_neurons=50):
        self.tau_m = tau_m
        self.tau_w = tau_w
        self.a = a
        self.b = b
        self.delta_T = delta_T
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = np.full(n_neurons, v_init)
        self.w = np.zeros(n_neurons)
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=n_neurons)

    def update(self, I, dt):
        dwdt = (self.a * (self.v - self.v_reset) - self.w) / self.tau_w
        self.w += dwdt * dt
        dvdt = (-self.v + self.delta_T * np.exp((self.v - self.v_th) / self.delta_T) + self.weights * self.w + I) / self.tau_m
        self.v += dvdt * dt
        
        spikes = self.v >= self.v_th
        self.v = np.where(spikes, self.v_reset, self.v)
        self.w = np.where(spikes, self.w + self.b, self.w)
        
        return spikes
