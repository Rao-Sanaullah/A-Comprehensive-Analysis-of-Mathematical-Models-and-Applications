import numpy as np

# define HH model
class HH:
    def __init__(self, v_init=-65.0, n_init=0.3177, m_init=0.0529, h_init=0.5961, n_neurons=50):
        self.v = v_init
        self.n = n_init
        self.m = m_init
        self.h = h_init
        self.n_neurons = n_neurons
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))
        self.C = 1.0
        self.g_Na = 120.0
        self.g_K = 36.0
        self.g_L = 0.3
        self.E_Na = 50.0
        self.E_K = -77.0
        self.E_L = -54.4
        self.num_ops = 18  # initial number of operations
        
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
        
        self.num_ops = 18 + 15  # number of operations for the above lines
        
        spike = self.v >= 0.0
        if spike:
            self.v = 30.0
            self.n = 0.3177
            self.m = 0.0529
            self.h = 0.5961
            self.num_ops += 32  # number of operations for the above lines
        self.num_ops += 12  # number of operations for the if condition and return statement
        
        return spike