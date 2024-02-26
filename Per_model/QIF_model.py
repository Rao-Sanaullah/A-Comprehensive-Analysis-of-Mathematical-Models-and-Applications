import numpy as np

# define QIF model
class QIF:
    def __init__(self, tau, v_reset, v_th, v_init=0.0, beta=0.5):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.beta = beta
        
    def update(self, I, dt):
        dvdt = (-self.v + self.beta * self.v**2 + I) / self.tau
        self.v += dvdt * dt
        if self.v >= self.v_th:
            spike = True
            self.v = self.v_reset
        else:
            spike = False
        return spike