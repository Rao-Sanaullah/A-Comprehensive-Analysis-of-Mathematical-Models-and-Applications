#GNU GENERAL PUBLIC LICENSE

# Copyright (C) Software Foundation, Inc. <https://fsf.org/>
# Only Author of this code is permitted to copy and distribute verbatim copies
# of this license document. Please contact us for contribution~!

import numpy as np

# define ThetaNeuron model
class ThetaNeuron:
    def __init__(self, tau, v_reset, v_th, v_init=0.0, n_neurons=50):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.n_neurons = n_neurons
        self.theta = np.random.uniform(low=0, high=2 * np.pi, size=(n_neurons, 1))
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))

    def update(self, I, dt):
        dthetadt = 1.0 - np.cos(self.theta)
        dvdt = (-self.v + I + np.dot(self.weights.T, dthetadt)) / self.tau
        self.v += dvdt * dt
        self.theta += 0.05 * 2 * np.pi * dt  # theta frequency is 10 Hz
        self.theta %= 2 * np.pi
        if self.v >= self.v_th:
            spike = True
            self.v = self.v_reset
        else:
            spike = False
        return spike
