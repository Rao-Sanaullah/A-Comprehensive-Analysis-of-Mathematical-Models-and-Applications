
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

n_samples = 1000
x1 = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
x2 = np.random.normal(loc=3.0, scale=1.0, size=n_samples)
X = np.hstack([x1, x2])
y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
from sklearn.utils import shuffle
X, y = shuffle(X.reshape(-1, 1), y)



######################################################### train LIF model
lif = LIF(tau=4, v_reset=0.0, v_th=1.0)
lif_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = lif.update(I, dt=1.0)
    lif_spikes[i] = spike

########################################################## train NLIF model
nlif = NLIF(tau=4, v_reset=0.0, v_th=1.0, alpha=0.5, beta=0.5)
nlif_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = nlif.update(I, dt=1.0)
    nlif_spikes[i] = spike


###################################################### train AdEX model
adex = AdEX(tau_m=4, v_rheo=0.5, v_spike=1.0, delta_T=1.0, v_reset=0.0)
adex_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = adex.update(I, dt=1.0)
    adex_spikes[i] = spike

##################################################### train HH model

hh = HH(v_init=-75.0, n_init=0.3177, m_init=0.0529, h_init=0.5961)
hh_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = hh.update(I, dt=1.0)
    hh_spikes[i] = spike

####################################################### train Izhikevich model

izh = Izhikevich(a=0.02, b=0.2, c=-65, d=6)
izh_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = izh.update(I, dt=1.0)
    izh_spikes[i] = spike

###################################################### train SRM model

srm = SRM(tau_s=8, tau_r=8, v_reset=0.0, v_th=1.0)
srm_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = srm.update(I, dt=1.0)
    srm_spikes[i] = spike

###################################################### train IFSFA model

ifsfa = IFSFA(tau_m=4, tau_w=100, a=0.1, b=0.01, delta_T=2, v_reset=0.0, v_th=1.0)
ifsfa_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = ifsfa.update(I, dt=1.0)
    ifsfa_spikes[i] = spike

##################################################### train QIF model

qif = QIF(tau=4, v_reset=0.0, v_th=1.0, beta=0.5)
qif_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = qif.update(I, dt=1.0)
    qif_spikes[i] = spike

##################################################### train ThetaNeuron model

theta = ThetaNeuron(tau=4, v_reset=0.0, v_th=1.0)
theta_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = theta.update(I, dt=1.0)
    theta_spikes[i] = spike

##################################################### compute classification accuracy

lif_acc = np.mean((lif_spikes == 1) == y)
nlif_acc = np.mean((nlif_spikes == 1) == y)
adex_acc = np.mean((adex_spikes == 1) == y)
hh_acc = np.mean((hh_spikes == 1) == y)
izh_acc = np.mean((izh_spikes == 1) == y)
srm_acc = np.mean((srm_spikes == 1) == y)
ifsfa_acc = np.mean((ifsfa_spikes == 1) == y)
qif_acc = np.mean((qif_spikes == 1) == y)
theta_acc = np.mean((theta_spikes == 1) == y)

print('LIF accuracy: {:.2f}%'.format(lif_acc * 100))
print('NLIF accuracy: {:.2f}%'.format(nlif_acc * 100))
print('AdEX accuracy: {:.2f}%'.format(adex_acc * 100))
print('HH accuracy: {:.2f}%'.format(hh_acc * 100))
print('Izhikevich accuracy: {:.2f}%'.format(izh_acc * 100))
print('SRM accuracy: {:.2f}%'.format(srm_acc * 100))
print('IF-SFA accuracy: {:.2f}%'.format(ifsfa_acc * 100))
print('QIF accuracy: {:.2f}%'.format(qif_acc * 100))
print('ThetaNeuron accuracy: {:.2f}%'.format(theta_acc * 100))
print('LIF Error Rate: {:.2f}%'.format(1 - lif_acc))
print('NLIF Error Rate: {:.2f}%'.format(1 - nlif_acc))
print('AdEX Error Rate: {:.2f}%'.format(1 - adex_acc))
print('HH Error Rate: {:.2f}%'.format(1 - hh_acc))
print('Izhikevich Error Rate: {:.2f}%'.format(1 - izh_acc))
print('SRM Error Rate: {:.2f}%'.format(1 - srm_acc))
print('IF-SFA Error Rate: {:.2f}%'.format(1 - ifsfa_acc))
print('QIF Error Rate: {:.2f}%'.format(1 - qif_acc))
print('ThetaNeuron Error Rate: {:.2f}%'.format(1 - theta_acc))

######################################################################### Combine 'Accuracy', 'Error' of all models

# set up the data and labels
x_labels = ['LIF', 'NLIF', 'AdEX', 'HH', 'Izh', 'SRM', 'IFSFA', 'QIF', 'ThetaNeuron']
colors = ['blue', 'orange', 'green', 'red', 'yellow', 'black', 'brown', 'purple', 'tan']
labels = ['Accuracy', 'Error']

data = {
    'LIF': {
        'Accuracy': lif_acc,
        'Error': 1 - lif_acc
    },
    'NLIF': {
        'Accuracy': nlif_acc,
        'Error': 1 - nlif_acc
    },
    'AdEX': {
        'Accuracy': adex_acc,
        'Error': 1 - adex_acc
    },
    'HH': {
        'Accuracy': hh_acc,
        'Error': 1 - hh_acc
    },
    'Izh': {
        'Accuracy': izh_acc,
        'Error': 1 - izh_acc
    },
    'SRM': {
        'Accuracy': srm_acc,
        'Error': 1 - srm_acc
    },
    'IFSFA': {
        'Accuracy': ifsfa_acc,
        'Error': 1 - ifsfa_acc 
    },
    'QIF': {
        'Accuracy': qif_acc,
        'Error': 1 - qif_acc 
    },
    'ThetaNeuron': {
        'Accuracy': theta_acc,
        'Error': 1 - theta_acc 
    }
}

fig, axs = plt.subplots(2, 1, figsize=(15, 8))
axs = axs.flatten()

for i, label in enumerate(labels):
    for j, x_label in enumerate(x_labels):
        y = data[x_label][label]
        color = colors[j]
        if label == 'Accuracy':
            marker = 'o'
        else:
            marker = 's'
        axs[i].plot(x_label, y, marker=marker, color=color, label=x_label)
    axs[i].set_title(label)
    axs[i].set_xlabel('Model')
    axs[i].set_ylabel('Percentage')
    axs[i].set_ylim(0, 1)
    axs[i].grid(True)
axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=10)
plt.rcParams.update({'font.size': 14})
fig.patch.set_facecolor('#f2f2f2')
fig.subplots_adjust(hspace=0.37)
plt.show()



