import numpy as np

class HH:
    def __init__(self, v_init=-65.0, n_init=0.3177, m_init=0.0529, h_init=0.5961):
        self.v = v_init
        self.n = n_init
        self.m = m_init
        self.h = h_init
        self.C = 1.0
        self.g_Na = 120.0
        self.g_K = 36.0
        self.g_L = 0.3
        self.E_Na = 50.0
        self.E_K = -77.0
        self.E_L = -54.4
        self.num_ops = 0  # initial number of operations
        
    def update(self, I, dt):
        alpha_n = (0.01 * (self.v + 55.0)) / (1.0 - np.exp(-0.1 * (self.v + 55.0)))
        beta_n = 0.125 * np.exp(-0.0125 * (self.v + 65.0))
        alpha_m = (0.1 * (self.v + 40.0)) / (1.0 - np.exp(-0.1 * (self.v + 40.0)))
        beta_m = 4.0 * np.exp(-0.0556 * (self.v + 65.0))
        alpha_h = 0.07 * np.exp(-0.05 * (self.v + 65.0))
        beta_h = 1.0 / (1.0 + np.exp(-0.1 * (self.v + 35.0)))
        
        # Intermediate computations
        n_inf = alpha_n / (alpha_n + beta_n)
        m_inf = alpha_m / (alpha_m + beta_m)
        h_inf = alpha_h / (alpha_h + beta_h)
        
        # Update gating variables
        self.n += (n_inf - self.n) * dt
        self.m += (m_inf - self.m) * dt
        self.h += (h_inf - self.h) * dt
        
        # Currents
        ina = self.g_Na * self.m**3 * self.h * (self.v - self.E_Na)
        ik = self.g_K * self.n**4 * (self.v - self.E_K)
        il = self.g_L * (self.v - self.E_L)
        
        # Update membrane potential
        dvdt = (I - ina - ik - il) / self.C
        self.v += dvdt * dt
        
        # Increment number of operations
        self.num_ops += 6  # Number of operations for intermediate computations
        
        # Check for spike
        spike = self.v >= -65.0
        if spike:
            self.v = 30.0
            self.n = 0.3177
            self.m = 0.0529
            self.h = 0.5961
            self.num_ops += 24  # Number of operations for resetting variables
        
        self.num_ops += 14  # Number of operations for other calculations and conditions
        
        return spike
