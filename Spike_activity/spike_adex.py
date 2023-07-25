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


#################################### Define AdEx neuron model
V_adex = np.zeros(len(t))
w_adex = np.zeros(len(t))
spikes_adex = np.zeros(len(t))
for i in range(1, len(t)):
    dV = (-V_adex[i-1] + Rm*I[i-1]/g_leak + w_adex[i-1]*np.exp((V_adex[i-1]-Vth)/tau)) / tau
    V_adex[i] = V_adex[i-1] + dV*dt
    dw = (a*(V_adex[i-1]-Vreset) - w_adex[i-1]) / tau_w
    w_adex[i] = w_adex[i-1] + dw*dt
    if V_adex[i] >= Vth:
        spikes_adex[i] = 0.9
        V_adex[i] = Vreset
        w_adex[i] += b


# Plot the results
plt.figure(figsize=(10,5))
plt.plot(t, V_adex, color='orange', label='Membrane potential')
plt.plot(t, spikes_adex, color='red', marker='x', linestyle='None', label='Spikes')
plt.title('AdEX neuron model', fontsize=16)
plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('Membrane potential (mV) / Spikes', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()