"""
Educational Use License

This software is provided solely for educational purposes. 
Any use of this software for commercial or non-educational purposes is prohibited. 
If you use this software in an academic setting, you must cite the following paper in any resulting publications or presentations:

[Sanaullah, Koravuna, Shamini, Ulrich Rückert, and Thorsten Jungeblut. "Exploring spiking neural networks: a comprehensive analysis of mathematical models and applications." Frontiers in Computational Neuroscience 17 (2023).]
[Sanaullah, Shamini Koravuna, Ulrich Rückert, and Thorsten Jungeblut. "Evaluating Spiking Neural Network Models: A Comparative Performance Analysis." Dataninja 2023 Annual Spring School(2023).]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to use the Software for educational purposes only, subject to the following conditions:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, 
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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


##################################### Create instances of each model

lif = LIF(tau=4, v_reset=0.0, v_th=1.0)
nlif = NLIF(tau=0.4, v_reset=0.0, v_th=1.0, alpha=0.9, beta=0.9)
adex = AdEX(tau_m=4, v_rheo=0.5, v_spike=1.0, delta_T=1.0, v_reset=0.0)
hh = HH(v_init=-75.0, n_init=0.3177, m_init=0.0529, h_init=0.5961)
izh = Izhikevich(a=0.02, b=2, c=-65, d=6)
srm = SRM(tau_s=0.3, tau_r=10, v_reset=0.0, v_th=1.0)
ifsfa = IFSFA(tau_m=4, tau_w=100, a=0.1, b=0.01, delta_T=2, v_reset=0.0, v_th=1.0)
qif = QIF(tau=4, v_reset=0.0, v_th=1.0, beta=0.5)
theta = ThetaNeuron(tau=4, v_reset=0.0, v_th=1.0)

T = 1000
X = np.random.normal(loc=0.0, scale=1.0, size=(T,))

models = [lif, nlif, adex, hh, ifsfa, qif, theta, izh, srm]
dt_values = [1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
spike_activities = [[] for _ in range(len(models))]

for i, x in enumerate(X):
    I = x
    for j, model in enumerate(models):
        spike_activities[j].append(model.update(I, dt=dt_values[j]))

for i, model in enumerate(models):
    print('Number of operations for {}: {}'.format(model.__class__.__name__, model.num_ops))

fig, axs = plt.subplots(nrows=len(models), ncols=1, sharex=True, figsize=(8, 10))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']

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
