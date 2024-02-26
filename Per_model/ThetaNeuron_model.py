import numpy as np

class ThetaNeuron:
    def __init__(self, tau, v_reset, v_th, v_init=0.0):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.theta = np.random.uniform(low=0, high=2 * np.pi)
        self.weights = np.random.normal(loc=0.0, scale=1.0)

    def update(self, I, dt):
        dthetadt = 1.0 - np.cos(self.theta)
        dvdt = (-self.v + I + np.dot(self.weights, dthetadt)) / self.tau
        self.v += dvdt * dt
        self.theta += 0.05 * 2 * np.pi * dt  # theta frequency is 10 Hz
        self.theta %= 2 * np.pi
        spike = False
        if self.v >= self.v_th:
            spike = True
            self.v = self.v_reset
        return spike
