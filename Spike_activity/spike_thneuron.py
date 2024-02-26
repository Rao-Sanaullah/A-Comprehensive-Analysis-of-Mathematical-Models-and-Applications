import numpy as np
import matplotlib.pyplot as plt

# Define parameters
dt = 0.1  # time step
t = np.arange(0, 1000, dt)  # time vector
V0 = -70  # initial potential
Vr = -70  # reset potential
theta = -50  # threshold potential
tau = 20  # membrane time constant
g = 1.5  # input conductance

# Define input current
I = np.zeros(len(t))
I[500:1500] = 10  # input current pulse

# Define Theta neuron model
V_theta = np.zeros(len(t))
spikes_theta = np.zeros(len(t))
for i in range(1, len(t)):
    dV = (-(V_theta[i-1] - Vr) + g*(theta - V_theta[i-1]) + I[i-1]) / tau
    V_theta[i] = V_theta[i-1] + dV*dt
    if V_theta[i] >= theta:
        spikes_theta[i] = 1
        V_theta[i] = Vr

# Plot the results
plt.figure(figsize=(10,5))
plt.plot(t, V_theta, color='darkgreen', label='Membrane potential')
plt.plot(t, spikes_theta, color='red', marker='x', linestyle='None', label='Spikes')
plt.title('Theta neuron model', fontsize=16)
plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('Membrane potential (mV) / Spikes', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()
