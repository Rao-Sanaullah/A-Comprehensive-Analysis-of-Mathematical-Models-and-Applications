import numpy as np

# define Izhikevich model
class Izhikevich:
    def __init__(self, a, b, c, d, v_init=0.1, u_init=0.2, n_neurons=50):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = np.full(n_neurons, v_init)
        self.u = np.full(n_neurons, u_init)
        
    def update(self, I, dt):
        dvdt = 0.04*self.v**2 + 5*self.v + 20 - self.u + I
        dudt = self.a*(self.b*self.v - self.u)
        self.v += dvdt * dt
        self.u += dudt * dt
        spike = self.v >= 0.8
        self.v = np.where(spike, self.c, self.v)
        self.u = np.where(spike, self.u + self.d, self.u)
        return spike
    

