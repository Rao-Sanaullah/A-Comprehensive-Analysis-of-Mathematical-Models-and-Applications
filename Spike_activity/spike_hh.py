
import numpy as np
import matplotlib.pyplot as plt



# Define parameters
Cm = 1  # membrane capacitance
g_Na = 120  # maximum sodium conductance
g_K = 36  # maximum potassium conductance
g_L = 0.3  # leak conductance
E_Na = 50  # sodium reversal potential
E_K = -82  # potassium reversal potential
E_L = -84.4  # leak reversal potential


# Define parameters
dt = 1.5  # time step
t = np.arange(0, 100, dt)  # time vector
Vreset = 0  # reset potential

# Define input current
I = np.zeros(len(t))
I[500:1500] = 0.5  # input current pulse

#################################################### Define HH neuron model

Vthh = -66  # spike threshold

V = np.zeros(len(t))
n = np.zeros(len(t))
m = np.zeros(len(t))
h = np.zeros(len(t))
spikes_hh = np.zeros(len(t))

# Initialize the state variables
V[0] = -65  # initial voltage
n[0] = 0.3177  # initial gating variable n
m[0] = 0.0529  # initial gating variable m
h[0] = 0.5961  # initial gating variable h

# Define the HH model equations
def alpha_n(V):
    return (0.01 * (V + 55)) / (1 - np.exp(-0.1 * (V + 55)))

def beta_n(V):
    return 0.125 * np.exp(-0.0125 * (V + 65))

def alpha_m(V):
    return (0.1 * (V + 40)) / (1 - np.exp(-0.1 * (V + 40)))

def beta_m(V):
    return 4 * np.exp(-0.0556 * (V + 65))

def alpha_h(V):
    return 0.07 * np.exp(-0.05 * (V + 65))

def beta_h(V):
    return 1 / (1 + np.exp(-0.1 * (V + 35)))

for i in range(1, len(t)):
    # Calculate the membrane currents
    I_Na = g_Na * m[i-1]**3 * h[i-1] * (V[i-1] - E_Na)
    I_K = g_K * n[i-1]**4 * (V[i-1] - E_K)
    I_L = g_L * (V[i-1] - E_L)
    I_m = I[i-1] / Cm  # input current

    # Update the state variables
    n[i] = n[i-1] + dt * (alpha_n(V[i-1]) * (1 - n[i-1]) - beta_n(V[i-1]) * n[i-1])
    m[i] = m[i-1] + dt * (alpha_m(V[i-1]) * (1 - m[i-1]) - beta_m(V[i-1]) * m[i-1])
    h[i] = h[i-1] + dt * (alpha_h(V[i-1]) * (1 - h[i-1]) - beta_h(V[i-1]) * h[i-1])
    V[i] = V[i-1] + dt * (I_m - I_Na - I_K - I_L) / Cm


    # Check for a spike
    if V[i] >= Vthh:
        spikes_hh[i] = 1 # spike detected
        V[i] = Vreset # reset voltage
        n[i] = alpha_n(V[i]) / (alpha_n(V[i]) + beta_n(V[i])) # reset gating variables
        m[i] = alpha_m(V[i]) / (alpha_m(V[i]) + beta_m(V[i]))
        h[i] = alpha_h(V[i]) / (alpha_h(V[i]) + beta_h(V[i]))



# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(t, V, color='olive', label='Membrane potential')
plt.plot(t, spikes_hh, color='red', marker='x', linestyle='None', label='Spikes')
plt.title('Hodgkin-Huxley neuron model', fontsize=16)
plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('Membrane potential (mV) / Spikes', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()

