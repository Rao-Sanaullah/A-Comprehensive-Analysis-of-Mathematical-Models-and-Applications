#GNU GENERAL PUBLIC LICENSE

# Copyright (C) Software Foundation, Inc. <https://fsf.org/>
# Only Author of this code is permitted to copy and distribute verbatim copies
# of this license document. Please contact us for contribution~!


#The accuracy is calculated as the mean of the element-wise equality comparison and then multiplied by 100 to obtain a percentage value. 
#A higher accuracy indicates a better-performing model in terms of its ability to correctly classify the input patterns.

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

##################################################### compute classification accuracy

lif_acc = np.mean(lif_spikes == y)
nlif_acc = np.mean(nlif_spikes == y)
adex_acc = np.mean(adex_spikes == y)
hh_acc = np.mean(hh_spikes == y)
izh_acc = np.mean(izh_spikes == y)
srm_acc = np.mean(srm_spikes == y)
ifsfa_acc = np.mean(ifsfa_spikes == y)
qif_acc = np.mean(qif_spikes == y)
theta_acc = np.mean(theta_spikes == y)

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





#################################################################### plot results
fig, axs = plt.subplots(4, 3, figsize=(12, 8))
axs[0, 0].scatter(X, y, c=lif_spikes, cmap='coolwarm', vmin=0, vmax=1)
axs[0, 0].set_title('LIF')
axs[0, 1].scatter(X, y, c=nlif_spikes, cmap='coolwarm', vmin=0, vmax=1)
axs[0, 1].set_title('NLIF')
axs[0, 2].scatter(X, y, c=adex_spikes, cmap='coolwarm', vmin=0, vmax=1)
axs[0, 2].set_title('AdEX')
axs[1, 0].scatter(X, y, c=hh_spikes, cmap='coolwarm', vmin=0, vmax=1)
axs[1, 0].set_title('HH')
axs[1, 1].scatter(X, y, c=izh_spikes, cmap='coolwarm', vmin=0, vmax=1)
axs[1, 1].set_title('Izhikevich')
axs[1, 2].scatter(X, y, c=srm_spikes, cmap='coolwarm', vmin=0, vmax=1)
axs[1, 2].set_title('SRM')
axs[2, 0].scatter(X, y, c=ifsfa_spikes, cmap='coolwarm', vmin=0, vmax=1)
axs[2, 0].set_title('IFSFA')
axs[2, 1].scatter(X, y, c=qif_spikes, cmap='coolwarm', vmin=0, vmax=1)
axs[2, 1].set_title('QIF')
axs[2, 2].scatter(X, y, c=theta_spikes, cmap='coolwarm', vmin=0, vmax=1)
axs[2, 2].set_title('ThetaNeuron')


axs[3, 1].plot(['LIF', 'NLIF', 'AdEX', 'HH', 'Izh', 'SRM', 'IFSFA', 'QIF', 'TNeu'], [lif_acc, nlif_acc, adex_acc, hh_acc, izh_acc, srm_acc, ifsfa_acc, qif_acc, theta_acc], '-o', color='blue')
axs[3, 1].plot(['LIF', 'NLIF', 'AdEX', 'HH', 'Izh', 'SRM', 'IFSFA', 'QIF', 'TNeu'], [1 - lif_acc, 1 - nlif_acc, 1 - adex_acc, 1 - hh_acc, 1 - izh_acc, 1 - srm_acc, 1- ifsfa_acc, 1- qif_acc, 1 - theta_acc], '-o', color='orange')
axs[3, 1].set_ylim(0, 1)
axs[3, 1].set_ylabel('Percentage')
axs[3, 1].set_title('Performance Comparison')
axs[3, 1].legend(['Accuracy', 'Error'])
plt.tight_layout()
plt.show()

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

# set up the figure and axes
fig, axs = plt.subplots(2, 1, figsize=(15, 8))
axs = axs.flatten()

# plot the data
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

# set the font size
plt.rcParams.update({'font.size': 14})

# set the background color
fig.patch.set_facecolor('#f2f2f2')

fig.subplots_adjust(hspace=0.37)

plt.show()



