import numpy as np

# define Izhikevich model
class Izhikevich:
    def __init__(self, a, b, c, d, v_init=-65.0, u_init=0.0, n_neurons=50):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = v_init
        self.u = u_init
        self.n_neurons = n_neurons
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))
        self.num_ops = 0  # initialize operation counter
        
    def update(self, I, dt):
        dvdt = 0.04*self.v**2 + 5*self.v + 140 - self.u + I
        dudt = self.a*(self.b*self.v - self.u)
        self.v += dvdt * dt
        self.u += dudt * dt
        spike = self.v >= 30.0
        self.v = np.where(spike, self.c, self.v)
        self.u = np.where(spike, self.u + self.d, self.u)
        
        if spike:
            self.num_ops += 4  # two multiplications, two additions
        else:
            self.num_ops += 2  # one multiplication, one addition
        
        self.num_ops += 4  # two multiplications, two additions for dvdt and dudt calculations
        
        return spike