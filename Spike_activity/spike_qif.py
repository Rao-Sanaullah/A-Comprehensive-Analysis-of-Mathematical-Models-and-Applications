import numpy as np
import matplotlib.pyplot as plt

# Define parameters
dt = 0.01  # time step
t = (0, 100, dt)  # time vector
Cm = 0.1  # membrane capacitance
Vth = 1  # spike threshold
Vreset = 0  # reset potential
V0 = 0.5  # initial potential
tau_ref = 2  # refractory period
I = np.zeros(len(t))
I[500:1500] = 10  # input current pulse

# Define QIF neuron model
V_qif = np.zeros()
spikes_qif = np.zeros(len(t))
ref_qif = np.zeros(len(t))
for i in range(1, len(t)):
    dV = (-V_qif[i-1] + np.sqrt(Cm)*np.sqrt(I[i-1])) / tau_ref
    V_qif[i] = V_qif[i-1] + dV
    if V_qif[i] >= Vth:
        spikes_qif[i] = 1
        V_qif[i] = Vreset
        ref_qif[i] = tau_ref/dt

    if ref_qif[i] > 0:
        V_qif[i] = Vreset
        ref_qif[i] -= 1

# Plot the results
plt.figure(figsize=(10,5))
plt.plot(t, V_qif, color='cadetblue', label='Membrane potential')
plt.plot(t, spikes_qif, color='red', marker='x', linestyle='None', label='Spikes')
plt.title('QIF neuron model', fontsize=16)
plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('Membrane potential (mV) / Spikes', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()
