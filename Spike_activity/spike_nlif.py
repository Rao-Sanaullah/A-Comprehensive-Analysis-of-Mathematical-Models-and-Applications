import numpy as np
import matplotlib.pyplot as plt

# Define parameters
dt = 0.1  # time step
t = np.arange(0, 100, dt)  # time vector
tau = 10  # membrane time constant
Rm = 1  # membrane resistance
Vth = 1  # spike threshold
Vreset = 0  # reset potential
V0 = 0.5  # initial potential
g_leak = 0.1  # leak conductance
tau_w = 30  # adaptation time constant
a = 0.001  # subthreshold adaptation conductance
b = 0.01  # spike-triggered adaptation increment

# Define input current
I = np.zeros(len(t))
I[500:1500] = 0.5  # input current pulse


####################################### Define NLIF neuron model
V_nlif = np.zeros(len(t))
spikes_nlif = np.zeros(len(t))
for i in range(1, len(t)):
    dV = (-V_nlif[i-1] + Rm*I[i-1]/g_leak + V_nlif[i-1]**2) / tau
    V_nlif[i] = V_nlif[i-1] + dV*dt
    if V_nlif[i] >= Vth:
        spikes_nlif[i] = 0.7
        V_nlif[i] = Vreset


# Plot the results
plt.figure(figsize=(10,5))
plt.plot(t, V_nlif, color='green', label='Membrane potential')
plt.plot(t, spikes_nlif, color='red', marker='x', linestyle='None', label='Spikes')
plt.title('NLIF neuron model', fontsize=16)
plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('Membrane potential (mV) / Spikes', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()




