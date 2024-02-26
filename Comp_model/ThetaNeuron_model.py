import numpy as np

# define ThetaNeuron model
class ThetaNeuron:
    def __init__(self, tau, v_reset, v_th, v_init=0.0):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.num_ops = 0  # initialize the number of operations to 0

    def update(self, I, dt):
        dthetadt = 1.0 - np.cos(self.theta)
        dvdt = (-self.v + I + np.dot(self.weights.T, dthetadt)) / self.tau
        self.v += dvdt * dt
        self.theta += 0.05 * 2 * np.pi * dt  # theta frequency is 10 Hz
        self.theta %= 2 * np.pi
        self.num_ops += 3  # increment the number of operations by 3 for each update (2 multiplications and 1 addition)
        if self.v >= self.v_th:
            spike = True
            self.v = self.v_reset
            self.num_ops += 1  # increment the number of operations by 1 for the spike
        else:
            spike = False
        return spike
