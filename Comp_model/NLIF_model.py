import numpy as np

# define NLIF model
class NLIF:
    def __init__(self, tau, v_reset, v_th, v_init=0.0, alpha=1.0, beta=1.0):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.alpha = alpha
        self.beta = beta
        self.num_ops = 5  # one multiplication, one subtraction, one comparison, two additions
        
    def update(self, I, dt):
        dvdt = (-self.v + I) / self.tau
        self.v += dvdt * dt
        spike = self.v >= self.v_th
        if spike:
            self.v = self.v_reset + self.alpha * (self.v - self.v_th)
            self.num_ops += 4  # two multiplications, two additions
        else:
            self.v *= self.beta
            self.num_ops += 1  # one multiplication
        self.num_ops += 4  # two subtractions, two comparisons
        return spike