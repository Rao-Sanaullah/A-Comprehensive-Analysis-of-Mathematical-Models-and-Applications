import numpy as np

# define IFSFA model
class IFSFA:
    def __init__(self, tau_m, tau_w, a, b, delta_T, v_reset, v_th, v_init=0.0, n_neurons=50):
        self.tau_m = tau_m
        self.tau_w = tau_w
        self.a = a
        self.b = b
        self.delta_T = delta_T
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.n_neurons = n_neurons
        self.w = np.zeros(n_neurons)
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))
        self.num_ops = 0
        
    def update(self, I, dt):
        dwdt = (self.a*(self.v-self.v_reset)-self.w) / self.tau_w
        self.w += dwdt * dt
        dvdt = (-self.v + self.delta_T*np.exp((self.v-self.v_th)/self.delta_T) + np.dot(self.weights.T, self.w) + I) / self.tau_m
        self.v += dvdt * dt
        self.num_ops += 2  # increment number of operations for the two derivative calculations above
        if self.v >= self.v_th:
            spike = True
            self.v = self.v_reset
            self.w += self.b
            self.num_ops += 2  # increment number of operations for the reset and synaptic update
        else:
            spike = False
        return spike