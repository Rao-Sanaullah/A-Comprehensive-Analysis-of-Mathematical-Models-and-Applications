
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
from Bio_model.LIF_model import LIF
from Bio_model.NLIF_model import NLIF
from Bio_model.AdEx_model import AdEX
from Bio_model.HH_model import HH
from Bio_model.IZH_model import Izhikevich
from Bio_model.SRM_model import SRM
from Bio_model.IFSFA_model import IFSFA
from Bio_model.QIF_model import QIF
from Bio_model.ThetaNeuron_model import ThetaNeuron


t = np.arange(0, 100, 0.1)
I = np.sin(t) + np.random.normal(scale=0.1, size=len(t))
lif = LIF(tau=4, v_reset=0.0, v_th=1.0, v_init=-0.1,n_neurons=1000)  
nlif = NLIF(tau=4, v_reset=0.0, v_th=1.0, v_init=-0.1, alpha=0.5, beta=0.5,n_neurons=1000)  
adex = AdEX(tau_m=4, v_rheo=0.5, v_spike=1.0, delta_T=1.0, v_reset=-0.1, v_init=-0.1,n_neurons=1000) 
ifsfa = IFSFA(tau_m=4, tau_w=100, a=0.1, b=0.01, delta_T=2, v_reset=0.0, v_th=1.0, v_init=-0.1,n_neurons=1000) 
qif = QIF(tau=4, v_reset=0.0, v_th=1.0, v_init=-0.1,beta=0.5,n_neurons=1000)  
theta = ThetaNeuron(tau=4, v_reset=0.0, v_th=1.0, v_init=-0.1,n_neurons=1000) 
srm = SRM(tau_s=0.3, tau_r=10, v_reset=0.0, v_th=1.0, v_init=0.0,n_neurons=1000)
izh = Izhikevich(a=0.02, b=0.2, c=0.1, d=0.06, v_init=0.01,u_init=0.2,n_neurons=1000)  

plt.figure(figsize=(8, 4))
plt.subplot(2, 1, 1)
plt.plot(t, I)
plt.ylabel('Input current')
plt.title("Compare the Models Biological Plausibility")

plt.subplot(2, 1, 2)
lif_v = []
nlif_v = []
adex_v = []
ifsfa_v = []
qif_v = []
theta_v = []
srm_v = []
izh_v = []

for i in range(len(t)):
    lif_spike = lif.update(I[i], 0.1)
    nlif_spike = nlif.update(I[i], 0.1)
    adex_spike = adex.update(I[i], 0.1)
    ifsfa_spike = ifsfa.update(I[i], 0.1)
    qif_spike = qif.update(I[i], 0.1)
    theta_spike = theta.update(I[i], 0.1)
    srm_spike = srm.update(I[i], 0.1)
    izh_spike = izh.update(I[i], 0.1)
    lif_v.append(lif.v)
    nlif_v.append(nlif.v)
    adex_v.append(adex.v)
    ifsfa_v.append(ifsfa.v)
    qif_v.append(qif.v)
    theta_v.append(theta.v)
    srm_v.append(srm.v)
    izh_v.append(izh.v)

plt.plot(t, lif_v)
plt.plot(t, nlif_v)
plt.plot(t, adex_v)
plt.plot(t, ifsfa_v)
plt.plot(t, qif_v)
plt.plot(t, theta_v)
plt.plot(t, srm_v)
plt.plot(t, izh_v)
plt.ylabel('Membrane potential')

plt.text(15, -2.5, 'LIF', color='red', ha='center')
plt.text(25, -2.5, 'NLIF', color='blue', ha='center')
plt.text(35, -2.5, 'AdEx', color='green', ha='center')
plt.text(45, -2.5, 'IFSFA', color='darkred', ha='center')
plt.text(55, -2.5, 'QIF', color='purple', ha='center')
plt.text(65, -2.5, 'Theta', color='darkblue', ha='center')
plt.text(75, -2.5, 'SRM', color='magenta', ha='center')
plt.text(85, -2.5, 'IZH', color='black', ha='center')

plt.show()


