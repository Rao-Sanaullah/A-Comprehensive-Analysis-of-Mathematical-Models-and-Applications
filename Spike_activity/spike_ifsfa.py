import numpy as np
import matplotlib.pyplot as plt

# Define parameters
dt = 0.1  # time step
t = np.arange(0, 100, dt)  # time vector

# Define parameters
Cm = 100  # membrane capacitance
vr = -60  # resting membrane potential
vt = -40  # threshold potential

k = 0.7  # sensitivity of the membrane potential

vpeak = 35  # spike cut-off

d = 100  # after-spike reset adaptation variable
c = -50  # after-spike reset membrane potential
b = -0.1  # spike-triggered adaptation increment
a = 0.01  # subthreshold adaptation conductance
tau_w = 200  # adaptation time constant

# Additional parameters
Rm = 1  # membrane resistance (in Ohms)

I = np.zeros(len(t))
I[500:1500] = 20  # input current pulse


################################# Define IFSFA neuron model

v_ifsfa = np.zeros(len(t))
u_ifsfa = np.zeros(len(t))
spikes_ifsfa = np.zeros(len(t))

for i in range(1, len(t)):
    dv = (Rm * (0.04 * v_ifsfa[i-1]**2 + 5 * v_ifsfa[i-1] + 140 - u_ifsfa[i-1] + I[i-1])) / Cm
    v_ifsfa[i] = v_ifsfa[i-1] + dv * dt
    du = a * (k * (v_ifsfa[i-1] - vr) - u_ifsfa[i-1])
    u_ifsfa[i] = u_ifsfa[i-1] + du * dt

    if v_ifsfa[i] >= vpeak:
        spikes_ifsfa[i] = 1
        v_ifsfa[i] = c
        u_ifsfa[i] = u_ifsfa[i-1] + d

    elif v_ifsfa[i] < vr:
        v_ifsfa[i] = vr

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(t, v_ifsfa, color='blue', label='Membrane potential')
plt.plot(t, spikes_ifsfa, color='red', marker='x', linestyle='None', label='Spikes')
plt.title('IFSFA neuron model', fontsize=16)
plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('Membrane potential (mV) / Spikes', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()



