
# examine how well the models mimic the behavior of real neurons
# or compare their predictions to experimental data
# Each model in this study has its own set of parameters that determine how the neuron behaves.

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


################################################## compare the models' biological plausibility

# for example, plot the membrane potential and firing rate of LIF and NLIF models
t = np.arange(0, 100, 0.1)
I = np.sin(t) + np.random.normal(scale=0.1, size=len(t))

# create instances of the models
lif = LIF(tau=4, v_reset=0.0, v_th=1.0, v_init=-0.1, n_neurons=1000)  # Set v_init to -0.1
nlif = NLIF(tau=4, v_reset=0.0, v_th=1.0, alpha=0.5, beta=0.5, v_init=-0.1, n_neurons=1000)  # Set v_init to -0.1
adex = AdEX(tau_m=4, v_rheo=0.5, v_spike=1.0, delta_T=1.0, v_reset=-0.1, v_init=-0.1, n_neurons=1000)  # Set v_init to -0.1
hh = HH(v_init=-75.0, n_init=0.3177, m_init=0.0529, h_init=0.5961, n_neurons=1000)
ifsfa = IFSFA(tau_m=4, tau_w=100, a=0.1, b=0.01, delta_T=2, v_reset=0.0, v_th=1.0, v_init=-0.1, n_neurons=1000)  # Set v_init to -0.1
qif = QIF(tau=4, v_reset=0.0, v_th=1.0, v_init=-0.1, n_neurons=1000)  # Set v_init to -0.1
theta = ThetaNeuron(tau=4, v_reset=0.0, v_th=1.0, v_init=-0.1, n_neurons=1000)  # Set v_init to -0.1
srm = SRM(tau_s=0.3, tau_r=10, v_reset=0.0, v_th=1.0, v_init=0.0, n_neurons=1000)
izh = Izhikevich(a=0.02, b=0.2, c=0.1, d=0.06, v_init=0.01, n_neurons=1000)  # Set v_init to -65.0



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
izh_spikes = []
srm_spikes = [] 
lif_spikes = []
nlif_spikes = []
adex_spikes = []
ifsfa_spikes = []
qif_spikes = []
theta_spikes = []
for i in range(len(t)):
    lif_spike = lif.update(I[i], 0.1)
    nlif_spike = nlif.update(I[i], 0.1)
    adex_spike = adex.update(I[i], 0.1)
    ifsfa_spike = ifsfa.update(I[i], 0.1)
    qif_spike = qif.update(I[i], 0.1)
    theta_spike = theta.update(I[i], 0.1)
    srm_spike = srm.update(I[i], 0.1)
    izh_spike = izh.update(I[i], 0.1)
    hh_spike = hh.update(I[i], 0.1)
    lif_spikes.append(lif_spike)
    nlif_spikes.append(nlif_spike)
    adex_spikes.append(adex_spike)
    ifsfa_spikes.append(ifsfa_spike)
    qif_spikes.append(qif_spike)
    theta_spikes.append(theta_spike)
    srm_spikes.append(srm_spike)
    izh_spikes.append(izh_spike)
    lif_v.append(lif.v)
    nlif_v.append(nlif.v)
    adex_v.append(adex.v)
    ifsfa_v.append(ifsfa.v)
    qif_v.append(qif.v)
    theta_v.append(theta.v)
    srm_v.append(srm.v)
    izh_v.append(izh.v)


plt.plot(t, lif_v, label='LIF')
plt.plot(t, nlif_v, label='NLIF')
plt.plot(t, adex_v, label='AdEx')
plt.plot(t, ifsfa_v, label='IFSFA')
plt.plot(t, qif_v, label='QIF')
plt.plot(t, theta_v, label='Theta')
plt.plot(t, srm_v, label='SRM')
plt.plot(t, izh_v, label='IZH')
plt.ylabel('Membrane potential')
plt.legend()


plt.show()

plt.figure(figsize=(8, 4))
plt.subplot(2, 1, 1)
plt.plot(t, lif_spikes, label='LIF')
plt.plot(t, nlif_spikes, label='NLIF')
plt.ylabel('Firing Rate')
plt.legend()

