class SRM:
    def __init__(self, tau_s, tau_r, v_reset, v_th, v_init=0.0):
        self.tau_s = tau_s
        self.tau_r = tau_r
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.s = 0.0
        self.r = 0.0
        self.weight = np.random.normal(loc=0.0, scale=1.0)
        self.num_ops = 3  

    def update(self, I, dt):
        dsdt = -self.s / self.tau_s + self.r
        drdt = -self.r / self.tau_r
        self.s += dsdt * dt
        self.r += drdt * dt
        dvdt = (-self.v + self.weight * self.s + I) / self.tau_s
        self.v += dvdt * dt
        self.num_ops += 3  

        if self.v >= self.v_th:
            spike = True
            self.v = self.v_reset
            self.s += 1.0
            self.r += 1.0
            self.num_ops += 5
        else:
            spike = False
            self.num_ops += 1  # one comparison

        return spike
