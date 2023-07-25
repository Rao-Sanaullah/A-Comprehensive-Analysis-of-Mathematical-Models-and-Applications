#GNU GENERAL PUBLIC LICENSE

# Copyright (C) Software Foundation, Inc. <https://fsf.org/>
# Only Author of this code is permitted to copy and distribute verbatim copies
# of this license document. Please contact us for contribution~!

import numpy as np

# define LIF model
class LIF:
    def __init__(self, tau, v_reset, v_th, v_init=0.0, n_neurons=50):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.n_neurons = n_neurons
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))

    def update(self, I, dt):
        dvdt = (-self.v + I) / self.tau
        self.v += dvdt * dt
        if self.v >= self.v_th:
            spike = True
            self.v = self.v_reset
        else:
            spike = False
        return spike
