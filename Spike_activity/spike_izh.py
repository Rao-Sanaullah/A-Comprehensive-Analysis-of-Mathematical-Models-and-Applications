import numpy as np
import matplotlib.pyplot as plt

# Define parameters
dt = 0.1  # time step
t = (0, 1000, dt)  # time vector
V0 = -90  # initial potential
Vr = -90  # reset potential
theta = -60  # threshold potential
tau = 20  # membrane time constant
a = 0.02  # recovery variable time scale
b = 0.2  # sensitivity of the recovery variable
c = -95  # after-spike reset potential
d = 6  # after-spike reset of the recovery variable

# Additional parameters
Rm = 10  # membrane resistance (in Ohms)
refractory_period = 10  # refractory period after a spike (in ms)

# Define input current
I = np.zeros(len(t))
I[500:1500] = 10  # input current pulse

# Define Izhikevich neuron model
V_izh = np.zeros(len(t))
u = np.zeros(len(t))
spikes_izh = np.zeros(len(t))
in_refractory_period = 0

for i in range(1):
    if in_refractory_period > 0:
        in_refractory_period -= 1
        V_izh[i] = Vr
        continue

    dv = (Rm * (0.04 * V_izh[i-1]**2 + 5 ( V_izh[i-1] )+ 140 - u[i-1] + I[i-1])) / tau
    du = a * (b * V_izh[i-1] - u[i-1])
    V_izh[i] = V_izh[i-1] + dv * dt
    u[i] = u[i-1] + du * dt

    if V_izh[i] >= theta:
        spikes_izh[i] = 1
        V_izh[i] = c
        u[i] = u[i] + d
        in_refractory_period = refractory_period / dt  # convert refractory period to time steps

# Print the spike activity
print("Spikes:", spikes_izh)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(t, V_izh, color='maroon', label='Membrane potential')
plt.plot(t, spikes_izh, color='red', marker='x', linestyle='None', label='Spikes')
plt.title('Izhikevich neuron model', fontsize=16)
plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('Membrane potential (mV) / Spikes', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()



