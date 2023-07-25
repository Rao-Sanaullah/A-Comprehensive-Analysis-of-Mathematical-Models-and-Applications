#GNU GENERAL PUBLIC LICENSE

# Copyright (C) Software Foundation, Inc. <https://fsf.org/>
# Only Author of this code is permitted to copy and distribute verbatim copies
# of this license document. Please contact us for contribution~!

"""

Performance loss, on the other hand, quantifies the deviation or error of the model’s predictions from the ground truth or desired output. 
It provides an indication of the model’s ability to accurately represent the input data.
In this study, the performance loss is calculated as the error rate, which demonstrates the percentage of misclassified samples. 
A lower error rate indicates a better-performing model with less deviation from the desired output.

It is important to note that the specific definitions and measurements for classification accuracy and performance loss
may vary depending on the context and objectives of the study.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from Per_model.LIF_model import LIF
from Per_model.NLIF_model import NLIF
from Per_model.AdEx_model import AdEX
from Per_model.HH_model import HH
from Per_model.IZH_model import Izhikevich
from Per_model.SRM_model import SRM
from Per_model.IFSFA_model import IFSFA
from Per_model.QIF_model import QIF
from Per_model.ThetaNeuron_model import ThetaNeuron


######################################################## creating Synthetic dataset

import numpy as np

# Create dataset
n_samples = 1000
x1 = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
x2 = np.random.normal(loc=3.0, scale=1.0, size=n_samples)
X = np.hstack([x1, x2])
y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

# Shuffle dataset using sklearn.utils.shuffle
from sklearn.utils import shuffle
X, y = shuffle(X.reshape(-1, 1), y)

# The shape of X is (2000, 1), and the shape of y is (2000,).
# You can reshape X to have it in the format (2000, 1) if required.
# X = X.reshape(-1, 1)


######################################################### train LIF model
lif = LIF(tau=4, v_reset=0.0, v_th=1.0, n_neurons=1000)
lif_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = lif.update(I, dt=1.0)
    lif_spikes[i] = spike

########################################################## train NLIF model
nlif = NLIF(tau=4, v_reset=0.0, v_th=1.0, alpha=0.5, beta=0.5, n_neurons=1000)
nlif_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = nlif.update(I, dt=1.0)
    nlif_spikes[i] = spike


###################################################### train AdEX model
adex = AdEX(tau_m=4, v_rheo=0.5, v_spike=1.0, delta_T=1.0, v_reset=0.0, n_neurons=1000)
adex_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = adex.update(I, dt=1.0)
    adex_spikes[i] = spike

##################################################### train HH model

hh = HH(v_init=-75.0, n_init=0.3177, m_init=0.0529, h_init=0.5961, n_neurons=1000)
hh_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = hh.update(I, dt=1.0)
    hh_spikes[i] = spike

####################################################### train Izhikevich model

izh = Izhikevich(a=0.02, b=0.2, c=-65, d=6, n_neurons=1000)
izh_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = izh.update(I, dt=1.0)
    izh_spikes[i] = spike

###################################################### train SRM model

srm = SRM(tau_s=8, tau_r=8, v_reset=0.0, v_th=1.0, n_neurons=1000)
srm_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = srm.update(I, dt=1.0)
    srm_spikes[i] = spike

###################################################### train IFSFA model

ifsfa = IFSFA(tau_m=4, tau_w=100, a=0.1, b=0.01, delta_T=2, v_reset=0.0, v_th=1.0, n_neurons=1000)
ifsfa_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = ifsfa.update(I, dt=1.0)
    ifsfa_spikes[i] = spike

##################################################### train QIF model

qif = QIF(tau=4, v_reset=0.0, v_th=1.0, n_neurons=1000, beta=0.5)
qif_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = qif.update(I, dt=1.0)
    qif_spikes[i] = spike

##################################################### train ThetaNeuron model

theta = ThetaNeuron(tau=4, v_reset=0.0, v_th=1.0, n_neurons=1000)
theta_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = theta.update(I, dt=1.0)
    theta_spikes[i] = spike

##################################################### compute classification

lif_acc = np.mean(lif_spikes == y)
nlif_acc = np.mean(nlif_spikes == y)
adex_acc = np.mean(adex_spikes == y)
hh_acc = np.mean(hh_spikes == y)
izh_acc = np.mean(izh_spikes == y)
srm_acc = np.mean(srm_spikes == y)
ifsfa_acc = np.mean(ifsfa_spikes == y)
qif_acc = np.mean(qif_spikes == y)
theta_acc = np.mean(theta_spikes == y)

###################################################### compute performance loss


##################################################################### mperformance loss of NLIF model relative to other models 


fig, ax = plt.subplots(figsize=(12, 8))

# plot 1
ax.plot(['LIF', 'NLIF'], [lif_acc, nlif_acc], '-o', color='green', label='LIF vs. NLIF Performance', markersize=8)
ax.plot(['LIF', 'NLIF'], [1 - lif_acc, 1 - nlif_acc], '-o', color='red', label='LIF vs. NLIF Error', markersize=8)

ax.plot(['LIF', 'AdEX'], [lif_acc, adex_acc], '-o', color='green', label='LIF vs. AdEX Performance', markersize=8)
ax.plot(['LIF', 'AdEX'], [1 - lif_acc, 1 - adex_acc], '-o', color='red', label='LIF vs. AdEX Error', markersize=8)

ax.plot(['LIF', 'HH'], [lif_acc, hh_acc], '-o', color='green', label='LIF vs. HH Performance', markersize=8)
ax.plot(['LIF', 'HH'], [1 - lif_acc, 1 - hh_acc], '-o', color='red', label='LIF vs. HH Error', markersize=8)

ax.plot(['LIF', 'IFSFA'], [lif_acc, ifsfa_acc], '-o', color='green', label='LIF vs. IFSFA Performance', markersize=8)
ax.plot(['LIF', 'IFSFA'], [1 - lif_acc, 1 - ifsfa_acc], '-o', color='red', label='LIF vs. IFSFA Error', markersize=8)

ax.plot(['LIF', 'QIF'], [lif_acc, qif_acc], '-o', color='green', label='LIF vs. QIF Performance', markersize=8)
ax.plot(['LIF', 'QIF'], [1 - lif_acc, 1 - qif_acc], '-o', color='red', label='LIF vs. QIF Error', markersize=8)

ax.plot(['LIF', 'ThNeuron'], [lif_acc, theta_acc], '-o', color='green', label='LIF vs. ThNeuron Performance', markersize=8)
ax.plot(['LIF', 'ThNeuron'], [1 - lif_acc, 1 - theta_acc ], '-o', color='red', label='LIF vs. ThNeuron Error', markersize=8)

ax.plot(['LIF', 'IZH'], [lif_acc, izh_acc], '-o', color='cyan', label='LIF vs. IZH Performance', markersize=8)
ax.plot(['LIF', 'IZH'], [1 - lif_acc, 1 - izh_acc], '-o', color='purple', label='LIF vs. IZH Error', markersize=8)

ax.plot(['LIF', 'SRM'], [lif_acc, srm_acc], '-o', color='cyan', label='LIF vs. SRM Performance', markersize=8)
ax.plot(['LIF', 'SRM'], [1 - lif_acc, 1 - srm_acc ], '-o', color='purple', label='LIF vs. SRM Error', markersize=8)

ax.set_ylim(0, 1)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('Performance Loss of LIF Compared to Other Neuron Models', fontsize=16)
ax.legend(fontsize=10)
plt.show()


##################################################################### mperformance loss of NLIF model relative to other models 


fig, ax = plt.subplots(figsize=(12, 8))

# plot 1
ax.plot(['NLIF', 'LIF'], [nlif_acc, lif_acc], '-o', color='cyan', label='NLIF vs. LIF Performance', markersize=8)
ax.plot(['NLIF', 'LIF'], [1 - nlif_acc, 1 - lif_acc], '-o', color='magenta', label='NLIF vs. LIF Error', markersize=8)

ax.plot(['NLIF', 'AdEX'], [nlif_acc, adex_acc], '-o', color='cyan', label='NLIF vs. AdEX Performance', markersize=8)
ax.plot(['NLIF', 'AdEX'], [1 - nlif_acc, 1 - adex_acc], '-o', color='magenta', label='NLIF vs. AdEX Error', markersize=8)

ax.plot(['NLIF', 'HH'], [nlif_acc, hh_acc], '-o', color='cyan', label='NLIF vs. HH Performance', markersize=8)
ax.plot(['NLIF', 'HH'], [1 - nlif_acc, 1 - hh_acc], '-o', color='magenta', label='NLIF vs. HH Error', markersize=8)

ax.plot(['NLIF', 'IFSFA'], [nlif_acc, ifsfa_acc], '-o', color='cyan', label='NLIF vs. IFSFA Performance', markersize=8)
ax.plot(['NLIF', 'IFSFA'], [1 - nlif_acc, 1 - ifsfa_acc], '-o', color='magenta', label='NLIF vs. IFSFA Error', markersize=8)

ax.plot(['NLIF', 'QIF'], [nlif_acc, qif_acc], '-o', color='cyan', label='NLIF vs. QIF Performance', markersize=8)
ax.plot(['NLIF', 'QIF'], [1 - nlif_acc, 1 - qif_acc], '-o', color='magenta', label='NLIF vs. QIF Error', markersize=8)

ax.plot(['NLIF', 'ThNeuron'], [nlif_acc, theta_acc], '-o', color='cyan', label='NLIF vs. ThNeuron Performance', markersize=8)
ax.plot(['NLIF', 'ThNeuron'], [1 - nlif_acc, 1 - theta_acc ], '-o', color='magenta', label='NLIF vs. ThNeuron Error', markersize=8)

ax.plot(['NLIF', 'IZH'], [nlif_acc, izh_acc], '-o', color='cyan', label='NLIF vs. IZH Performance', markersize=8)
ax.plot(['NLIF', 'IZH'], [1 - nlif_acc, 1 - izh_acc], '-o', color='purple', label='NLIF vs. IZH Error', markersize=8)

ax.plot(['NLIF', 'SRM'], [nlif_acc, srm_acc], '-o', color='cyan', label='NLIF vs. SRM Performance', markersize=8)
ax.plot(['NLIF', 'SRM'], [1 - nlif_acc, 1 - srm_acc ], '-o', color='purple', label='NLIF vs. SRM Error', markersize=8)

ax.set_ylim(0, 1)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('Performance Loss of NLIF Compared to Other Neuron Models', fontsize=16)
ax.legend(fontsize=10)
plt.show()

##################################################################### mperformance loss of AdEx model relative to other models 

fig, ax = plt.subplots(figsize=(12, 8))

# plot 1
ax.plot(['AdEX', 'LIF'], [adex_acc, lif_acc], '-o', color='blue', label='AdEX vs. LIF Performance', markersize=8)
ax.plot(['AdEX', 'LIF'], [1 - adex_acc, 1 - lif_acc], '-o', color='orange', label='AdEX vs. LIF Error', markersize=8)

ax.plot(['AdEX', 'NLIF'], [adex_acc, nlif_acc], '-o', color='blue', label='AdEX vs. NLIF Performance', markersize=8)
ax.plot(['AdEX', 'NLIF'], [1 - adex_acc, 1 - nlif_acc], '-o', color='orange', label='AdEX vs. NLIF Error', markersize=8)

ax.plot(['AdEX', 'HH'], [adex_acc, hh_acc], '-o', color='blue', label='AdEX vs. HH Performance', markersize=8)
ax.plot(['AdEX', 'HH'], [1 - adex_acc, 1 - hh_acc], '-o', color='orange', label='AdEX vs. HH Error', markersize=8)

ax.plot(['AdEX', 'IFSFA'], [adex_acc, ifsfa_acc], '-o', color='blue', label='AdEX vs. IFSFA Performance', markersize=8)
ax.plot(['AdEX', 'IFSFA'], [1 - adex_acc, 1 - ifsfa_acc], '-o', color='orange', label='AdEX vs. IFSFA Error', markersize=8)

ax.plot(['AdEX', 'QIF'], [adex_acc, qif_acc], '-o', color='blue', label='AdEX vs. QIF Performance', markersize=8)
ax.plot(['AdEX', 'QIF'], [1 - adex_acc, 1 - qif_acc], '-o', color='orange', label='AdEX vs. QIF Error', markersize=8)

ax.plot(['AdEX', 'ThNeuron'], [adex_acc, theta_acc], '-o', color='blue', label='AdEX vs. ThNeuron Performance', markersize=8)
ax.plot(['AdEX', 'ThNeuron'], [1 - adex_acc, 1 - theta_acc ], '-o', color='orange', label='AdEX vs. ThNeuron Error', markersize=8)

ax.plot(['AdEX', 'IZH'], [adex_acc, izh_acc], '-o', color='blue', label='AdEX vs. IZH Performance', markersize=8)
ax.plot(['AdEX', 'IZH'], [1 - adex_acc, 1 - izh_acc], '-o', color='purple', label='AdEX vs. IZH Error', markersize=8)

ax.plot(['AdEX', 'SRM'], [adex_acc, srm_acc], '-o', color='blue', label='AdEX vs. SRM Performance', markersize=8)
ax.plot(['AdEX', 'SRM'], [1 - adex_acc, 1 - srm_acc ], '-o', color='purple', label='AdEX vs. SRM Error', markersize=8)

ax.set_ylim(0, 1)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('Performance Loss of AdEX Compared to Other Neuron Models', fontsize=16)
ax.legend(fontsize=10)
plt.show()


##################################################################### mperformance loss of HH model relative to other models 

fig, ax = plt.subplots(figsize=(12, 8))

# plot 1
ax.plot(['HH', 'LIF'], [hh_acc, lif_acc], '-o', color='tan', label='HH vs. LIF Performance', markersize=8)
ax.plot(['HH', 'LIF'], [1 - hh_acc, 1 - lif_acc], '-o', color='purple', label='HH vs. LIF Error', markersize=8)

ax.plot(['HH', 'NLIF'], [hh_acc, nlif_acc], '-o', color='tan', label='HH vs. NLIF Performance', markersize=8)
ax.plot(['HH', 'NLIF'], [1 - hh_acc, 1 - nlif_acc], '-o', color='purple', label='HH vs. NLIF Error', markersize=8)

ax.plot(['HH', 'AdEX'], [hh_acc, adex_acc], '-o', color='tan', label='HH vs. AdEX Performance', markersize=8)
ax.plot(['HH', 'AdEX'], [1 - hh_acc, 1 - adex_acc], '-o', color='purple', label='HH vs. AdEX Error', markersize=8)

ax.plot(['HH', 'IFSFA'], [hh_acc, ifsfa_acc], '-o', color='tan', label='HH vs. IFSFA Performance', markersize=8)
ax.plot(['HH', 'IFSFA'], [1 - hh_acc, 1 - ifsfa_acc], '-o', color='purple', label='HH vs. IFSFA Error', markersize=8)

ax.plot(['HH', 'QIF'], [hh_acc, qif_acc], '-o', color='tan', label='HH vs. QIF Performance', markersize=8)
ax.plot(['HH', 'QIF'], [1 - hh_acc, 1 - qif_acc], '-o', color='purple', label='HH vs. QIF Error', markersize=8)

ax.plot(['HH', 'ThNeuron'], [hh_acc, theta_acc], '-o', color='tan', label='HH vs. ThNeuron Performance', markersize=8)
ax.plot(['HH', 'ThNeuron'], [1 - hh_acc, 1 - theta_acc ], '-o', color='purple', label='HH vs. ThNeuron Error', markersize=8)

ax.plot(['HH', 'IZH'], [hh_acc, izh_acc], '-o', color='tan', label='HH vs. IZH Performance', markersize=8)
ax.plot(['HH', 'IZH'], [1 - hh_acc, 1 - izh_acc], '-o', color='purple', label='HH vs. IZH Error', markersize=8)

ax.plot(['HH', 'SRM'], [hh_acc, srm_acc], '-o', color='tan', label='HH vs. SRM Performance', markersize=8)
ax.plot(['HH', 'SRM'], [1 - hh_acc, 1 - srm_acc ], '-o', color='purple', label='HH vs. SRM Error', markersize=8)

ax.set_ylim(0, 1)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('Performance Loss of HH Compared to Other Neuron Models', fontsize=16)
ax.legend(fontsize=10)
plt.show()

##################################################################### mperformance loss of QIF model relative to other models 

fig, ax = plt.subplots(figsize=(12, 8))

# plot 1
ax.plot(['QIF', 'LIF'], [qif_acc, lif_acc], '-o', color='brown', label='QIF vs. LIF Performance', markersize=8)
ax.plot(['QIF', 'LIF'], [1 - qif_acc, 1 - lif_acc], '-o', color='m', label='QIF vs. LIF Error', markersize=8)

ax.plot(['QIF', 'NLIF'], [qif_acc, nlif_acc], '-o', color='brown', label='QIF vs. NLIF Performance', markersize=8)
ax.plot(['QIF', 'NLIF'], [1 - qif_acc, 1 - nlif_acc], '-o', color='m', label='QIF vs. NLIF Error', markersize=8)

ax.plot(['QIF', 'AdEX'], [qif_acc, adex_acc], '-o', color='brown', label='QIF vs. AdEX Performance', markersize=8)
ax.plot(['QIF', 'AdEX'], [1 - qif_acc, 1 - adex_acc], '-o', color='m', label='QIF vs. AdEX Error', markersize=8)

ax.plot(['QIF', 'HH'], [qif_acc, hh_acc], '-o', color='brown', label='QIF vs. HH Performance', markersize=8)
ax.plot(['QIF', 'HH'], [1 - qif_acc, 1 - hh_acc], '-o', color='purple', label='QIF vs. HH Error', markersize=8)

ax.plot(['QIF', 'IFSFA'], [qif_acc, ifsfa_acc], '-o', color='brown', label='QIF vs. IFSFA Performance', markersize=8)
ax.plot(['QIF', 'IFSFA'], [1 - qif_acc, 1 - ifsfa_acc], '-o', color='m', label='QIF vs. IFSFA Error', markersize=8)

ax.plot(['QIF', 'ThNeuron'], [qif_acc, theta_acc], '-o', color='brown', label='QIF vs. ThNeuron Performance', markersize=8)
ax.plot(['QIF', 'ThNeuron'], [1 - qif_acc, 1 - theta_acc ], '-o', color='m', label='QIF vs. ThNeuron Error', markersize=8)

ax.plot(['QIF', 'IZH'], [qif_acc, izh_acc], '-o', color='brown', label='QIF vs. IZH Performance', markersize=8)
ax.plot(['QIF', 'IZH'], [1 - qif_acc, 1 - izh_acc], '-o', color='purple', label='QIF vs. IZH Error', markersize=8)

ax.plot(['QIF', 'SRM'], [qif_acc, srm_acc], '-o', color='brown', label='QIF vs. SRM Performance', markersize=8)
ax.plot(['QIF', 'SRM'], [1 - qif_acc, 1 - srm_acc ], '-o', color='purple', label='QIF vs. SRM Error', markersize=8)

ax.set_ylim(0, 1)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('Performance Loss of QIF Compared to Other Neuron Models', fontsize=16)
ax.legend(fontsize=10)
plt.show()


##################################################################### mperformance loss of SRM model relative to other models 

fig, ax = plt.subplots(figsize=(12, 8))

# plot 1
ax.plot(['SRM', 'LIF'], [srm_acc, lif_acc], '-o', color='g', label='SRM vs. LIF Performance', markersize=8)
ax.plot(['SRM', 'LIF'], [1 - srm_acc, 1 - lif_acc], '-o', color='c', label='SRM vs. LIF Error', markersize=8)

ax.plot(['SRM', 'NLIF'], [srm_acc, nlif_acc], '-o', color='g', label='SRM vs. NLIF Performance', markersize=8)
ax.plot(['SRM', 'NLIF'], [1 - srm_acc, 1 - nlif_acc], '-o', color='c', label='SRM vs. NLIF Error', markersize=8)

ax.plot(['SRM', 'AdEX'], [srm_acc, adex_acc], '-o', color='g', label='SRM vs. AdEX Performance', markersize=8)
ax.plot(['SRM', 'AdEX'], [1 - srm_acc, 1 - adex_acc], '-o', color='c', label='SRM vs. AdEX Error', markersize=8)

ax.plot(['SRM', 'HH'], [srm_acc, hh_acc], '-o', color='g', label='SRM vs. HH Performance', markersize=8)
ax.plot(['SRM', 'HH'], [1 - srm_acc, 1 - hh_acc], '-o', color='c', label='SRM vs. HH Error', markersize=8)

ax.plot(['SRM', 'QIF'], [srm_acc, qif_acc], '-o', color='g', label='SRM vs. QIF Performance', markersize=8)
ax.plot(['SRM', 'QIF'], [1 - srm_acc, 1 - qif_acc], '-o', color='c', label='SRM vs. QIF Error', markersize=8)

ax.plot(['SRM', 'IFSFA'], [srm_acc, ifsfa_acc], '-o', color='g', label='SRM vs. IFSFA Performance', markersize=8)
ax.plot(['SRM', 'IFSFA'], [1 - srm_acc, 1 - ifsfa_acc ], '-o', color='c', label='SRM vs. IFSFA Error', markersize=8)

ax.plot(['SRM', 'ThNeuron'], [srm_acc, theta_acc], '-o', color='g', label='SRM vs. ThNeuron Performance', markersize=8)
ax.plot(['SRM', 'ThNeuron'], [1 - srm_acc, 1 - theta_acc ], '-o', color='c', label='SRM vs. ThNeuron Error', markersize=8)

ax.plot(['SRM', 'IZH'], [srm_acc, izh_acc], '-o', color='g', label='SRM vs. IZH Performance', markersize=8)
ax.plot(['SRM', 'IZH'], [1 - srm_acc, 1 - izh_acc ], '-o', color='c', label='SRM vs. IZH Error', markersize=8)


ax.set_ylim(0, 1)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('Performance Loss of SRM Compared to Other Neuron Models', fontsize=16)
ax.legend(fontsize=10)
plt.show()


##################################################################### mperformance loss of Izhikevich model relative to other models 


fig, ax = plt.subplots(figsize=(12, 8))

# plot 1
ax.plot(['IZH', 'LIF'], [izh_acc, lif_acc], '-o', color='g', label='IZH vs. LIF Performance', markersize=8)
ax.plot(['IZH', 'LIF'], [1 - izh_acc, 1 - lif_acc], '-o', color='c', label='IZH vs. LIF Error', markersize=8)

ax.plot(['IZH', 'NLIF'], [izh_acc, nlif_acc], '-o', color='g', label='IZH vs. NLIF Performance', markersize=8)
ax.plot(['IZH', 'NLIF'], [1 - izh_acc, 1 - nlif_acc], '-o', color='c', label='IZH vs. NLIF Error', markersize=8)

ax.plot(['IZH', 'AdEX'], [izh_acc, adex_acc], '-o', color='g', label='IZH vs. AdEX Performance', markersize=8)
ax.plot(['IZH', 'AdEX'], [1 - izh_acc, 1 - adex_acc], '-o', color='c', label='IZH vs. AdEX Error', markersize=8)

ax.plot(['IZH', 'HH'], [izh_acc, hh_acc], '-o', color='g', label='IZH vs. HH Performance', markersize=8)
ax.plot(['IZH', 'HH'], [1 - izh_acc, 1 - hh_acc], '-o', color='c', label='IZH vs. HH Error', markersize=8)

ax.plot(['IZH', 'QIF'], [izh_acc, qif_acc], '-o', color='g', label='IZH vs. QIF Performance', markersize=8)
ax.plot(['IZH', 'QIF'], [1 - izh_acc, 1 - qif_acc], '-o', color='c', label='IZH vs. QIF Error', markersize=8)

ax.plot(['IZH', 'IFSFA'], [izh_acc, ifsfa_acc], '-o', color='g', label='IZH vs. IFSFA Performance', markersize=8)
ax.plot(['IZH', 'IFSFA'], [1 - izh_acc, 1 - ifsfa_acc ], '-o', color='c', label='IZH vs. IFSFA Error', markersize=8)

ax.plot(['IZH', 'ThNeuron'], [izh_acc, theta_acc], '-o', color='g', label='IZH vs. ThNeuron Performance', markersize=8)
ax.plot(['IZH', 'ThNeuron'], [1 - izh_acc, 1 - theta_acc ], '-o', color='c', label='IZH vs. ThNeuron Error', markersize=8)

ax.plot(['IZH', 'SRM'], [izh_acc, srm_acc], '-o', color='g', label='IZH vs. SRM Performance', markersize=8)
ax.plot(['IZH', 'SRM'], [1 - izh_acc, 1 - srm_acc ], '-o', color='c', label='IZH vs. SRM Error', markersize=8)

ax.set_ylim(0, 1)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('Performance Loss of Izhikevich Compared to Other Neuron Models', fontsize=16)
ax.legend(fontsize=10)
plt.show()


##################################################################### mperformance loss of ThNeuron model relative to other models 


fig, ax = plt.subplots(figsize=(12, 8))

# plot 1
ax.plot(['ThNeuron', 'LIF'], [theta_acc, lif_acc], '-o', color='green', label='ThNeuron vs. LIF Performance', markersize=8)
ax.plot(['ThNeuron', 'LIF'], [1 - theta_acc, 1 - lif_acc], '-o', color='c', label='ThNeuron vs. LIF Error', markersize=8)

ax.plot(['ThNeuron', 'NLIF'], [theta_acc, nlif_acc], '-o', color='green', label='ThNeuron vs. NLIF Performance', markersize=8)
ax.plot(['ThNeuron', 'NLIF'], [1 - theta_acc, 1 - nlif_acc], '-o', color='c', label='ThNeuron vs. NLIF Error', markersize=8)

ax.plot(['ThNeuron', 'AdEX'], [theta_acc, adex_acc], '-o', color='green', label='ThNeuron vs. AdEX Performance', markersize=8)
ax.plot(['ThNeuron', 'AdEX'], [1 - theta_acc, 1 - adex_acc], '-o', color='c', label='ThNeuron vs. AdEX Error', markersize=8)

ax.plot(['ThNeuron', 'HH'], [theta_acc, hh_acc], '-o', color='green', label='ThNeuron vs. HH Performance', markersize=8)
ax.plot(['ThNeuron', 'HH'], [1 - theta_acc, 1 - hh_acc], '-o', color='c', label='ThNeuron vs. HH Error', markersize=8)

ax.plot(['ThNeuron', 'QIF'], [theta_acc, qif_acc], '-o', color='green', label='ThNeuron vs. QIF Performance', markersize=8)
ax.plot(['ThNeuron', 'QIF'], [1 - theta_acc, 1 - qif_acc], '-o', color='c', label='ThNeuron vs. QIF Error', markersize=8)

ax.plot(['ThNeuron', 'IFSFA'], [theta_acc, ifsfa_acc], '-o', color='green', label='ThNeuron vs. IFSFA Performance', markersize=8)
ax.plot(['ThNeuron', 'IFSFA'], [1 - theta_acc, 1 - ifsfa_acc ], '-o', color='c', label='ThNeuron vs. IFSFA Error', markersize=8)

ax.plot(['ThNeuron', 'Izhikevich'], [theta_acc, izh_acc], '-o', color='green', label='ThNeuron vs. Izhikevich Performance', markersize=8)
ax.plot(['ThNeuron', 'Izhikevich'], [1 - theta_acc, 1 - izh_acc ], '-o', color='c', label='ThNeuron vs. Izhikevich Error', markersize=8)

ax.plot(['ThNeuron', 'SRM'], [theta_acc, srm_acc], '-o', color='green', label='ThNeuron vs. SRM Performance', markersize=8)
ax.plot(['ThNeuron', 'SRM'], [1 - theta_acc, 1 - srm_acc ], '-o', color='c', label='ThNeuron vs. SRM Error', markersize=8)

ax.set_ylim(0, 1)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('Performance Loss of ThNeuron Compared to Other Neuron Models', fontsize=16)
ax.legend(fontsize=10)
plt.show()


##################################################################### mperformance loss of IFSFA model relative to other models 


fig, ax = plt.subplots(figsize=(12, 8))

# plot 1
ax.plot(['IFSFA', 'LIF'], [ifsfa_acc, lif_acc], '-o', color='tan', label='IFSFA vs. LIF Performance', markersize=8)
ax.plot(['IFSFA', 'LIF'], [1 - ifsfa_acc, 1 - lif_acc], '-o', color='purple', label='IFSFA vs. LIF Error', markersize=8)

ax.plot(['IFSFA', 'NLIF'], [ifsfa_acc, nlif_acc], '-o', color='tan', label='IFSFA vs. NLIF Performance', markersize=8)
ax.plot(['IFSFA', 'NLIF'], [1 - ifsfa_acc, 1 - nlif_acc], '-o', color='purple', label='IFSFA vs. NLIF Error', markersize=8)

ax.plot(['IFSFA', 'AdEX'], [ifsfa_acc, adex_acc], '-o', color='tan', label='IFSFA vs. AdEX Performance', markersize=8)
ax.plot(['IFSFA', 'AdEX'], [1 - ifsfa_acc, 1 - adex_acc], '-o', color='purple', label='IFSFA vs. AdEX Error', markersize=8)

ax.plot(['IFSFA', 'HH'], [ifsfa_acc, hh_acc], '-o', color='tan', label='IFSFA vs. HH Performance', markersize=8)
ax.plot(['IFSFA', 'HH'], [1 - ifsfa_acc, 1 - hh_acc], '-o', color='purple', label='IFSFA vs. HH Error', markersize=8)

ax.plot(['IFSFA', 'QIF'], [ifsfa_acc, qif_acc], '-o', color='tan', label='IFSFA vs. QIF Performance', markersize=8)
ax.plot(['IFSFA', 'QIF'], [1 - ifsfa_acc, 1 - qif_acc], '-o', color='purple', label='IFSFA vs. QIF Error', markersize=8)

ax.plot(['IFSFA', 'ThNeuron'], [ifsfa_acc, theta_acc], '-o', color='tan', label='IFSFA vs. ThNeuron Performance', markersize=8)
ax.plot(['IFSFA', 'ThNeuron'], [1 - ifsfa_acc, 1 - theta_acc ], '-o', color='purple', label='IFSFA vs. ThNeuron Error', markersize=8)

ax.plot(['IFSFA', 'IZH'], [ifsfa_acc, izh_acc], '-o', color='tan', label='IFSFA vs. IZH Performance', markersize=8)
ax.plot(['IFSFA', 'IZH'], [1 - ifsfa_acc, 1 - izh_acc], '-o', color='purple', label='IFSFA vs. IZH Error', markersize=8)

ax.plot(['IFSFA', 'SRM'], [ifsfa_acc, srm_acc], '-o', color='tan', label='IFSFA vs. SRM Performance', markersize=8)
ax.plot(['IFSFA', 'SRM'], [1 - ifsfa_acc, 1 - srm_acc ], '-o', color='purple', label='IFSFA vs. SRM Error', markersize=8)

ax.set_ylim(0, 1)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('Performance Loss of IFSFA Compared to Other Neuron Models', fontsize=16)
ax.legend(fontsize=10)
plt.show()

