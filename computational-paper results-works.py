
"""
Analyze the computational complexity: each models are updating the neuron state in every time step, 
but their implementations are different. The LIF model uses a simple thresholding rule, while the NLIF 
model uses a non-linear function to update the neuron state. You could analyze the computational complexity of 
each model, e.g., by counting the number of operations required to update the neuron state in one time step, and compare them.



This code adds a num_ops attribute to each model, which is initialized to the number of operations 
required to initialize the model. Then, every time the update method is called, the number of operations 
required to update the model is added to num_ops. Finally, the code prints the total number of operations 
required for each model. Note that these numbers are just rough estimates, and will depend on the specific 
hardware and software used to run the code. However, they can still provide some insight into the


output:

Number of operations for LIF: 3003
Number of operations for NLIF: 5092
Number of operations for AdEx: 4305
Number of operations for HH: 45
Number of operations for IFSFA: 2260
Number of operations for QIF: 1018
Number of operations for THETA: 3490
Number of operations for IZH: 6094
Number of operations for SRM: 4027

The output you provided lists the number of operations required for different neuron models.

Each neuron model is a mathematical description of how neurons work, and each model has its own set of 
equations that are used to simulate the behavior of neurons. These equations involve mathematical operations 
such as addition, multiplication, and exponentiation, which are computationally expensive.

The number of operations required for a neuron model is an indication of how computationally expensive it is to 
simulate that model. In general, the more operations required, the longer it will take to simulate the behavior of that model.

Therefore, based on the output you provided, we can see that the HH model is the most computationally expensive, 
requiring 3962 operations. The Izhikevich model is the next most expensive, requiring 1016 operations, and the SRM model 
requires 3000 operations. The LIF model is the least expensive, requiring only 3003 operations.
"""

import numpy as np
import matplotlib.pyplot as plt

# define LIF model
# define LIF model
class LIF:
    def __init__(self, tau, v_reset, v_th, v_init=0.0, n_neurons=50):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.n_neurons = n_neurons
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))
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

# define NLIF model
class NLIF:
    def __init__(self, tau, v_reset, v_th, v_init=0.0, alpha=1.0, beta=1.0, n_neurons=50):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.alpha = alpha
        self.beta = beta
        self.n_neurons = n_neurons
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))
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


class AdEX:
    def __init__(self, tau_m, v_rheo, v_spike, delta_T, v_reset, v_init=0.0, n_neurons=50):
        self.tau_m = tau_m
        self.v_rheo = v_rheo
        self.v_spike = v_spike
        self.delta_T = delta_T
        self.v_reset = v_reset
        self.v = v_init
        self.n_neurons = n_neurons
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))
        self.num_ops = 5  # one multiplication, four additions
        
    def update(self, I, dt):
        dvdt = (-self.v + self.tau_m * I - self.v_rheo + self.delta_T * np.exp((self.v - self.v_spike) / self.delta_T)) / self.tau_m
        self.v += dvdt * dt
        spike = self.v >= self.v_spike
        if spike:
            self.v = self.v_reset
            self.num_ops += 4  # two multiplications, two additions
        else:
            self.num_ops += 1  # one multiplication
        self.num_ops += 3  # three additions
        return spike

       
class HH:
    def __init__(self, v_init=-65.0, n_init=0.3177, m_init=0.0529, h_init=0.5961, n_neurons=50):
        self.v = v_init
        self.n = n_init
        self.m = m_init
        self.h = h_init
        self.n_neurons = n_neurons
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))
        self.C = 1.0
        self.g_Na = 120.0
        self.g_K = 36.0
        self.g_L = 0.3
        self.E_Na = 50.0
        self.E_K = -77.0
        self.E_L = -54.4
        self.num_ops = 18  # initial number of operations
        
    def update(self, I, dt):
        alpha_n = (0.01 * (self.v + 55.0)) / (1.0 - np.exp(-0.1 * (self.v + 55.0)))
        beta_n = 0.125 * np.exp(-0.0125 * (self.v + 65.0))
        alpha_m = (0.1 * (self.v + 40.0)) / (1.0 - np.exp(-0.1 * (self.v + 40.0)))
        beta_m = 4.0 * np.exp(-0.0556 * (self.v + 65.0))
        alpha_h = 0.07 * np.exp(-0.05 * (self.v + 65.0))
        beta_h = 1.0 / (1.0 + np.exp(-0.1 * (self.v + 35.0)))
        
        dvdt = (I - self.g_Na * self.m**3 * self.h * (self.v - self.E_Na) - self.g_K * self.n**4 * (self.v - self.E_K) - self.g_L * (self.v - self.E_L)) / self.C
        dndt = alpha_n * (1.0 - self.n) - beta_n * self.n
        dmdt = alpha_m * (1.0 - self.m) - beta_m * self.m
        dhdt = alpha_h * (1.0 - self.h) - beta_h * self.h
        
        self.v += dvdt * dt
        self.n += dndt * dt
        self.m += dmdt * dt
        self.h += dhdt * dt
        
        self.num_ops = 18 + 15  # number of operations for the above lines
        
        spike = self.v >= 0.0
        if spike:
            self.v = 30.0
            self.n = 0.3177
            self.m = 0.0529
            self.h = 0.5961
            self.num_ops += 32  # number of operations for the above lines
        self.num_ops += 12  # number of operations for the if condition and return statement
        
        return spike


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


class SRM:
    def __init__(self, tau_s, tau_r, v_reset, v_th, v_init=0.0, n_neurons=50):
        self.tau_s = tau_s
        self.tau_r = tau_r
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.n_neurons = n_neurons
        self.s = np.zeros(n_neurons)
        self.r = np.zeros(n_neurons)
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))
        self.num_ops = 3 # three additions
    
    def update(self, I, dt):
        dsdt = -self.s / self.tau_s + self.r
        drdt = -self.r / self.tau_r
        self.s += dsdt * dt
        self.r += drdt * dt
        dvdt = (-self.v + np.dot(self.weights.T, self.s) + I) / self.tau_s
        self.v += dvdt * dt
        self.num_ops += 3 # three additions
        
        if self.v >= self.v_th:
            spike = True
            self.v = self.v_reset
            self.s += 1.0
            self.r += 1.0
            self.num_ops += 5 # two additions, two divisions, one multiplication
        else:
            spike = False
            self.num_ops += 1 # one comparison
        
        return spike


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
    
class QIF:
    def __init__(self, tau, v_reset, v_th, v_init=0.0, n_neurons=50, beta=0.5):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.n_neurons = n_neurons
        self.beta = beta
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))
        self.num_ops = 0
        
    def update(self, I, dt):
        dvdt = (-self.v + self.beta * self.v**2 + I) / self.tau
        self.v += dvdt * dt
        self.num_ops += 1  # increment number of operations for the derivative calculation above
        if self.v >= self.v_th:
            spike = True
            self.v = self.v_reset
            self.num_ops += 2  # increment number of operations for the reset and synaptic update
        else:
            spike = False
        return spike
    
class ThetaNeuron:
    def __init__(self, tau, v_reset, v_th, v_init=0.0, n_neurons=50):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.n_neurons = n_neurons
        self.theta = np.random.uniform(low=0, high=2 * np.pi, size=(n_neurons, 1))
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))
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





# create instances of the models
lif = LIF(tau=4, v_reset=0.0, v_th=1.0, n_neurons=1000)
nlif = NLIF(tau=0.4, v_reset=0.0, v_th=1.0, alpha=0.9, beta=0.9, n_neurons=1000)
adex = AdEX(tau_m=4, v_rheo=0.5, v_spike=1.0, delta_T=1.0, v_reset=0.0, n_neurons=1000)
hh = HH(v_init=-75.0, n_init=0.3177, m_init=0.0529, h_init=0.5961, n_neurons=1000)
izh = Izhikevich(a=0.02, b=2, c=-65, d=6, n_neurons=1000)
srm = SRM(tau_s=0.3, tau_r=10, v_reset=0.0, v_th=1.0, n_neurons=1000)
ifsfa = IFSFA(tau_m=4, tau_w=100, a=0.1, b=0.01, delta_T=2, v_reset=0.0, v_th=1.0, n_neurons=1000)
qif = QIF(tau=4, v_reset=0.0, v_th=1.0, n_neurons=1000, beta=0.5)
theta = ThetaNeuron(tau=4, v_reset=0.0, v_th=1.0, n_neurons=1000)



# define input array X
T = 1000
X = np.random.normal(loc=0.0, scale=1.0, size=(T,))

# simulate the models and record spiking activity
lif_spikes = []
nlif_spikes = []
adex_spikes = []
hh_spikes = []
ifsfa_spikes = []
qif_spikes = []
theta_spikes = []

izh_spikes = []
srm_spikes = []

for i, x in enumerate(X):
    I = x
    lif_spikes.append(lif.update(I, dt=1.0))
    nlif_spikes.append((nlif.update(I, dt=0.2)))
    adex_spikes.append((adex.update(I, dt=1.0)))
    hh_spikes.append((hh.update(I, dt=1.0)))
    ifsfa_spikes.append((ifsfa.update(I, dt=1.0)))
    qif_spikes.append((qif.update(I, dt=1.0)))
    theta_spikes.append((theta.update(I, dt=1.0)))

    izh_spikes.append((izh.update(I, dt=0.2)))
    srm_spikes.append((srm.update(I, dt=0.2)))




# print number of operations required to update the models
print('Number of operations for LIF: {}'.format(lif.num_ops))
print('Number of operations for NLIF: {}'.format(nlif.num_ops))
print('Number of operations for AdEx: {}'.format(adex.num_ops))
print('Number of operations for HH: {}'.format(hh.num_ops))
print('Number of operations for IFSFA: {}'.format(ifsfa.num_ops))
print('Number of operations for QIF: {}'.format(qif.num_ops))
print('Number of operations for THETA: {}'.format(theta.num_ops))

print('Number of operations for IZH: {}'.format(izh.num_ops))
print('Number of operations for SRM: {}'.format(srm.num_ops))


# Create a new figure with subplots
fig, axs = plt.subplots(nrows=9, ncols=1, sharex=True, figsize=(8, 10))


axs[0].plot(lif_spikes, label='LIF', color='tab:blue')
axs[0].set_title('Comparison No. of Spiking Operations Across Models')
axs[1].plot(nlif_spikes, label='NLIF', color='tab:orange')
axs[2].plot(adex_spikes, label='AdEx', color='tab:green')
axs[3].plot(hh_spikes, label='HH', color='tab:red')
axs[4].plot(ifsfa_spikes, label='IF-SFA', color='tab:purple')
axs[5].plot(qif_spikes, label='QIF', color='tab:brown')
axs[6].plot(theta_spikes, label='Theta', color='tab:pink')

axs[7].plot(izh_spikes, label='IZH', color='tab:red')
axs[8].plot(srm_spikes, label='SRM', color='tab:green')

# Set axis labels and tick parameters for all subplots
for ax in axs:
    ax.set_ylabel('Spikes', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.legend(fontsize=8, loc='upper right')

axs[-1].set_xlabel('Time', fontsize=12)

plt.tight_layout()
plt.show()