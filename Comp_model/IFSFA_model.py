import numpy as np

class IFSFA:
    def __init__(self, tau_m, tau_w, a, b, delta_T, v_reset, v_th, v_init=0.0):
        self.tau_m = tau_m
        self.tau_w = tau_w
        self.a = a
        self.b = b
        self.delta_T = delta_T
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.w = 0.0
        self.weight = np.random.normal(loc=0.0, scale=1.0)
        self.num_ops = 0

    def update(self, I, dt):
        dwdt = (self.a * (self.v - self.v_reset) - self.w) / self.tau_w
        self.w += dwdt * dt
        dvdt = (-self.v + self.delta_T * np.exp((self.v - self.v_th) / self.delta_T) + self.weight * self.w + I) / self.tau_m
        self.v += dvdt * dt
        if self.v >= self.v_th:
            spike = True
            self.v = self.v_reset
            self.w += self.b
            self.num_ops += 4 
        else:
            spike = False
            self.num_ops += 2  
        return spike
