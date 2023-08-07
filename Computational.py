#GNU GENERAL PUBLIC LICENSE

# Copyright (C) Software Foundation, Inc. <https://fsf.org/>
# Only Author of this code is permitted to copy and distribute verbatim copies
# of this license document. Please contact us for contribution~!

############# The output of this code provides lists of the number of operations required for each ne#################
"""
Each neuron model is a mathematical description of how neurons work, and each model has its own set of 
equations that are used to simulate the behavior of neurons. These equations involve mathematical operations 
such as addition, multiplication, and exponentiation, which are computationally expensive.

This code adds a num_ops attribute to each model, which is initialized to the number of operations 
required to initialize the model. Then, every time the update method is called, the number of operations 
required to update the model is added to num_ops. Finally, the code prints the total number of operations 
required for each model. Note that these numbers are just rough estimates, and will depend on the specific 
hardware and software used to run the code. However, they can still provide some insight into the

"""

import numpy as np
import matplotlib.pyplot as plt
from Comp_model.LIF_model import LIF
from Comp_model.NLIF_model import NLIF
from Comp_model.AdEx_model import AdEX
from Comp_model.HH_model import HH
from Comp_model.IZH_model import Izhikevich
from Comp_model.SRM_model import SRM
from Comp_model.IFSFA_model import IFSFA
from Comp_model.QIF_model import QIF
from Comp_model.ThetaNeuron_model import ThetaNeuron


# Create instances of each model

# LIF Model
lif = LIF(tau=4, v_reset=0.0, v_th=1.0, n_neurons=1000)
# NLIF Model
nlif = NLIF(tau=0.4, v_reset=0.0, v_th=1.0, alpha=0.9, beta=0.9, n_neurons=1000)
# AdEx Model
adex = AdEX(tau_m=4, v_rheo=0.5, v_spike=1.0, delta_T=1.0, v_reset=0.0, n_neurons=1000)
# Hodgkin-Huxley Model
hh = HH(v_init=-75.0, n_init=0.3177, m_init=0.0529, h_init=0.5961, n_neurons=1000)
# Izhikevich Model
izh = Izhikevich(a=0.02, b=2, c=-65, d=6, n_neurons=1000)
# Spike Response Model (SRM)
srm = SRM(tau_s=0.3, tau_r=10, v_reset=0.0, v_th=1.0, n_neurons=1000)
# Integrate-and-Fire with Spike-Frequency Adaptation (IFSFA) Model
ifsfa = IFSFA(tau_m=4, tau_w=100, a=0.1, b=0.01, delta_T=2, v_reset=0.0, v_th=1.0, n_neurons=1000)
# Quadratic Integrate-and-Fire (QIF) Model
qif = QIF(tau=4, v_reset=0.0, v_th=1.0, n_neurons=1000, beta=0.5)
# Theta Neuron Model
theta = ThetaNeuron(tau=4, v_reset=0.0, v_th=1.0, n_neurons=1000)

# Define input array X
T = 1000
X = np.random.normal(loc=0.0, scale=1.0, size=(T,))

# Simulate each model with specified dt_values
models = [lif, nlif, adex, hh, ifsfa, qif, theta, izh, srm]
dt_values = [1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
spike_activities = [[] for i in range(len(models))]

for i, x in enumerate(X):
    I = x
    for j, model in enumerate(models):
        spike_activities[j].(model.update(I, dt=dt_values[j]))

# Print number of operations required to update the models
for i, model in enumerate(models):
    print('Number of operations for {}: {}'.format(model.__name__, model.num_ops))

# Create a figure with subplots and specify colors for each plot
fig, axs = plt.subplots(nrows=len(models), ncols=1, sharex=True, figsize=(8, 10))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']

# Plot spiking activity for each model with the specified colors
for i, model in enumerate(models):
    axs[i].plot(spike_activities[i], label=model.__class__.__name__, color=colors[i])
    axs[i].set_ylabel('Spikes', fontsize=10)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].tick_params(axis='both', which='both', length=0)
    axs[i].legend(fontsize=8, loc='upper right')

axs[-1].set_xlabel('Time', fontsize=12)

plt.tight_layout()
plt.show()
