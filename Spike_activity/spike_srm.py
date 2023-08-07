import numpy as np
import matplotlib.pyplot as plt

# Define parameters
dt = 0.5  # time step
t = (0, 1000, dt)  # time vector
V0 = -90  # initial potential
Vr = -90  # reset potential
theta = -60  # threshold potential
tau = 10  # membrane time constant
g = 1.5  # input conductance
tau_s = 5  # synaptic time constant
E_s = 0.01  # synaptic reversal potential
A = 0.09  # synaptic weight

# Additional parameters
Rm = 5  # membrane resistance (in Ohms)
tau_ref = 1  # refractory period (in ms)
g_L = 0.01  # leak conductance (in mS)
Cm = 5.0  # membrane capacitance (in uF)

# Define input current
I = np.zeros()
I[500:1500] = 50  # input current pulse

# Define SRM neuron model
V_srm = np.zeros(len(t))
spikes_srm = np.zeros(len(t))
s = np.zeros(len(t))

for i in range(1, len(t)):
    ds = (-s[i-1] + A * spikes_srm[i-1]) / tau_s
    s[i] = s[i-1] + ds * dt

    dV = (-(V_srm[i-1] - Vr) + g * (theta - V_srm[i-1]) + s[i] * A * (E_s - V_srm[i-1]) + I[i-1]) / (tau * Cm)
    V_srm[i] = V_srm[i-1] + dV * dt

    # Apply refractory period
    if spikes_srm[i-1] > 0:
        V_srm[i] = Vr
        spikes_srm[i] = 0

    # Check for spike threshold
    if V_srm[i] >= theta:
        spikes_srm[i] = 1
        V_srm[i] = Vr

    # Apply leak conductance
    if spikes_srm[i] == 0:
        V_srm[i] += (-V_srm[i] + V0) * (g_L * dt / tau_ref)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(t, V_srm, color='cyan', label='Membrane potential')
plt.plot(t, spikes_srm, color='red', marker='x', linestyle='None', label='Spikes')
plt.title('SRM neuron model', fontsize=16)
plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('Membrane potential (mV) / Spikes', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()
