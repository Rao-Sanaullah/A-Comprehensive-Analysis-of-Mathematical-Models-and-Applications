import numpy as np

class HH:
    def __init__(self, n_neurons=50, v_init=-65.0, n_init=0.3177, m_init=0.0529, h_init=0.5961):
        self.v = np.full(n_neurons, v_init)
        self.n = np.full(n_neurons, n_init)
        self.m = np.full(n_neurons, m_init)
        self.h = np.full(n_neurons, h_init)
        self.C = np.full(n_neurons, 1.0)
        self.g_Na = np.full(n_neurons, 120.0)
        self.g_K = np.full(n_neurons, 36.0)
        self.g_L = np.full(n_neurons, 0.3)
        self.E_Na = np.full(n_neurons, 50.0)
        self.E_K = np.full(n_neurons, -77.0)
        self.E_L = np.full(n_neurons, -54.4)
        self.n_neurons = n_neurons

    def update(self, I, dt):
        alpha_n = (0.01 * (self.v + 55.0)) / (1.0 - np.exp(-0.1 * (self.v + 55.0)))
        beta_n = 0.125 * np.exp(-0.0125 * (self.v + 65.0))
        alpha_m = (0.1 * (self.v + 40.0)) / (1.0 - np.exp(-0.1 * (self.v + 40.0)))
        beta_m = 4.0 * np.exp(-0.0556 * (self.v + 65.0))
        alpha_h = 0.07 * np.exp(-0.05 * (self.v + 65.0))
        beta_h = 1.0 / (1.0 + np.exp(-0.1 * (self.v + 35.0)))
        
        dvdt = (I - self.g_Na * self.m**3 * self.h * (self.v - self.E_Na) - self.g_K * self.n**4 * (self.v - self.E_K) - self.g_L * (self.v - self.E_L)) / self.C
        dndt = alpha_n * (1.0 - self.n) - beta_n * self.n
        dmdt = alpha_m * (1.0 - self.m) - beta_m * self.m
        dhdt = alpha_h * (1.0 - self.h) - beta_h * self.h
        
        self.v += dvdt * dt
        self.n += dndt * dt
        self.m += dmdt * dt
        self.h += dhdt * dt
        
        spikes = self.v >= 0
        self.v = np.where(spikes, 30.0, self.v)
        self.n = np.where(spikes, 0.3177, self.n)
        self.m = np.where(spikes, 0.0529, self.m)
        self.h = np.where(spikes, 0.5961, self.h)
        
        return spikes
