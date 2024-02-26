import numpy as np

# define LIF model
class LIF:
    def __init__(self, tau, v_reset, v_th, v_init=0.0):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.num_ops = 3  # one multiplication, one subtraction, one comparison
        
    def update(self, I, dt):
        dvdt = (-self.v + I) / self.tau
        self.v += dvdt * dt
        if self.v >= self.v_th:
            spike = True
            self.v = self.v_reset
        else:
            spike = False
        self.num_ops += 3  # update number of operations
        return spike