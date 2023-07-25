#GNU GENERAL PUBLIC LICENSE

# Copyright (C) Software Foundation, Inc. <https://fsf.org/>
# Only Author of this code is permitted to copy and distribute verbatim copies
# of this license document. Please contact us for contribution~!

#the performance of the different SNN neural models was explored using classification accuracy and performance loss as the performance metrics, 
#where classification accuracy measures the percentage of correctly classified samples or data points by a model. 
#Therefore, we used classification accuracy to evaluate how accurately each SNN model classified the input data,
#based on the comparison of predicted spikes with the true labels.

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

#################################################################### compute performance loss of LIF relative to ThetaNeuron
perf_loss_lif_theta = (theta_acc - lif_acc) / theta_acc
print('Performance loss of LIF relative to ThetaNeuron: {:.2f}%'.format(perf_loss_lif_theta * 100))

# compute performance loss of LIF relative to NLIF
perf_loss_lif_nlif = (nlif_acc - lif_acc) / nlif_acc
print('Performance loss of LIF relative to NLIF: {:.2f}%'.format(perf_loss_lif_nlif * 100))

# compute performance loss of LIF relative to AdEx
perf_loss_lif_adex = (adex_acc - lif_acc) / adex_acc
print('Performance loss of LIF relative to AdEx: {:.2f}%'.format(perf_loss_lif_adex * 100))

# compute performance loss of LIF relative to HH
perf_loss_lif_hh = (hh_acc - lif_acc) / hh_acc
print('Performance loss of LIF relative to HH: {:.2f}%'.format(perf_loss_lif_hh * 100))

# compute performance loss of LIF relative to Izhikevich
perf_loss_lif_izh = (izh_acc - lif_acc) / izh_acc
print('Performance loss of LIF relative to Izhikevich: {:.2f}%'.format(perf_loss_lif_izh * 100))

# compute performance loss of LIF relative to HH
perf_loss_lif_srm = (srm_acc - lif_acc) / srm_acc
print('Performance loss of LIF relative to SRM: {:.2f}%'.format(perf_loss_lif_srm * 100))

# compute performance loss of LIF relative to IFSFA
perf_loss_lif_ifsfa = (ifsfa_acc - lif_acc) / ifsfa_acc
print('Performance loss of LIF relative to IFSFA: {:.2f}%'.format(perf_loss_lif_ifsfa * 100))

# compute performance loss of LIF relative to SRM
perf_loss_lif_qif = (qif_acc - lif_acc) / qif_acc
print('Performance loss of LIF relative to QIF: {:.2f}%'.format(perf_loss_lif_qif * 100))



#################################################################### compute performance loss of NLIF relative to ThetaNeuron
perf_loss_nlif_theta = (theta_acc - nlif_acc) / theta_acc
print('Performance loss of NLIF relative to ThetaNeuron: {:.2f}%'.format(perf_loss_nlif_theta * 100))

# compute performance loss of NLIF relative to NLIF
perf_loss_nlif_nlif = (lif_acc - nlif_acc) / lif_acc
print('Performance loss of NLIF relative to LIF: {:.2f}%'.format(perf_loss_lif_nlif * 100))

# compute performance loss of NLIF relative to AdEx
perf_loss_nlif_adex = (adex_acc - lif_acc) / adex_acc
print('Performance loss of NLIF relative to AdEx: {:.2f}%'.format(perf_loss_nlif_adex * 100))

# compute performance loss of NLIF relative to HH
perf_loss_nlif_hh = (hh_acc - lif_acc) / hh_acc
print('Performance loss of NLIF relative to HH: {:.2f}%'.format(perf_loss_nlif_hh * 100))

# compute performance loss of NLIF relative to Izhikevich
perf_loss_nlif_izh = (izh_acc - lif_acc) / izh_acc
print('Performance loss of NLIF relative to Izhikevich: {:.2f}%'.format(perf_loss_nlif_izh * 100))

# compute performance loss of NLIF relative to HH
perf_loss_nlif_srm = (srm_acc - lif_acc) / srm_acc
print('Performance loss of NLIF relative to SRM: {:.2f}%'.format(perf_loss_nlif_srm * 100))

# compute performance loss of NLIF relative to IFSFA
perf_loss_nlif_ifsfa = (ifsfa_acc - lif_acc) / ifsfa_acc
print('Performance loss of NLIF relative to IFSFA: {:.2f}%'.format(perf_loss_nlif_ifsfa * 100))

# compute performance loss of NLIF relative to SRM
perf_loss_nlif_qif = (qif_acc - lif_acc) / qif_acc
print('Performance loss of NLIF relative to QIF: {:.2f}%'.format(perf_loss_nlif_qif * 100))


########################################################################## compute performance loss of AdEX relative to LIF
perf_loss_adex_lif = (lif_acc - adex_acc) / lif_acc
print('Performance loss relative to AdEX with LIF: {:.2f}%'.format(perf_loss_adex_lif * 100))

# compute performance loss relative to AdEX with NLIF
perf_loss_adex_nlif = (nlif_acc - adex_acc) / nlif_acc
print('Performance loss relative to AdEX with NLIF: {:.2f}%'.format(perf_loss_adex_nlif * 100))

# compute performance loss relative to AdEX with ThetaNeuron
perf_loss_adex_theta = (theta_acc - adex_acc) / theta_acc
print('Performance loss of AdEX relative to ThetaNeuron: {:.2f}%'.format(perf_loss_adex_theta * 100))

# compute performance loss of AdEX relative to HH
perf_loss_adex_hh = (hh_acc - adex_acc) / hh_acc
print('Performance loss of AdEX relative to HH: {:.2f}%'.format(perf_loss_adex_hh * 100))

# compute performance loss of AdEX relative to Izhikevich
perf_loss_adex_izh = (izh_acc - adex_acc) / izh_acc
print('Performance loss of AdEX relative to Izhikevich: {:.2f}%'.format(perf_loss_adex_izh * 100))

# compute performance loss of AdEX relative to HH
perf_loss_adex_srm = (srm_acc - adex_acc) / srm_acc
print('Performance loss of AdEX relative to SRM: {:.2f}%'.format(perf_loss_adex_srm * 100))

# compute performance loss of AdEX relative to IFSFA
perf_loss_adex_ifsfa = (ifsfa_acc - adex_acc) / ifsfa_acc
print('Performance loss of AdEX relative to IFSFA: {:.2f}%'.format(perf_loss_adex_ifsfa * 100))

# compute performance loss of AdEX relative to SRM
perf_loss_adex_qif = (qif_acc - adex_acc) / qif_acc
print('Performance loss of AdEX relative to QIF: {:.2f}%'.format(perf_loss_adex_qif * 100))



################################################ compute performance loss of HH relative to LIF
perf_loss_hh_lif = (lif_acc - hh_acc) / lif_acc
print('Performance loss of HH relative to LIF: {:.2f}%'.format(perf_loss_hh_lif * 100))

# compute performance loss relative to HH to NLIF
perf_loss_hh_nlif = (nlif_acc - hh_acc) / nlif_acc
print('Performance loss of HH relative to NLIF: {:.2f}%'.format(perf_loss_hh_nlif * 100))

# compute performance loss of HH relative to AdEx
perf_loss_hh_adex = (adex_acc - hh_acc) / adex_acc
print('Performance loss of HH relative to AdEx: {:.2f}%'.format(perf_loss_hh_adex * 100))

# compute performance loss relative to HH with ThetaNeuron
perf_loss_hh_theta = (theta_acc - hh_acc) / theta_acc
print('Performance loss of HH relative to ThetaNeuron: {:.2f}%'.format(perf_loss_hh_theta * 100))

# compute performance loss of HH relative to Izhikevich
perf_loss_hh_izh = (izh_acc - hh_acc) / izh_acc
print('Performance loss of HH relative to Izhikevich: {:.2f}%'.format(perf_loss_hh_izh * 100))

# compute performance loss of HH relative to HH
perf_loss_hh_srm = (srm_acc - hh_acc) / srm_acc
print('Performance loss of HH relative to SRM: {:.2f}%'.format(perf_loss_hh_srm * 100))

# compute performance loss of HH relative to IFSFA
perf_loss_hh_ifsfa = (ifsfa_acc - hh_acc) / ifsfa_acc
print('Performance loss of HH relative to IFSFA: {:.2f}%'.format(perf_loss_hh_ifsfa * 100))

# compute performance loss of HH relative to SRM
perf_loss_hh_qif = (qif_acc - hh_acc) / qif_acc
print('Performance loss of HH relative to QIF: {:.2f}%'.format(perf_loss_hh_qif * 100))

################################################################## compute performance loss of Izhikevich relative to LIF
perf_loss_izh_lif = (lif_acc - izh_acc) / lif_acc
print('Performance loss of Izhikevich relative to LIF: {:.2f}%'.format(perf_loss_izh_lif * 100))

# compute performance loss of Izhikevich relative to NLIF
perf_loss_izh_nlif = (nlif_acc - izh_acc) / nlif_acc
print('Performance loss of Izhikevich relative to NLIF: {:.2f}%'.format(perf_loss_izh_nlif * 100))

# compute performance loss of Izhikevich relative to AdEx
perf_loss_izh_adex = (adex_acc - izh_acc) / adex_acc
print('Performance loss of Izhikevich relative to AdEx: {:.2f}%'.format(perf_loss_izh_adex * 100))

# compute performance loss of Izhikevich relative to HH
perf_loss_izh_hh = (hh_acc - izh_acc) / hh_acc
print('Performance loss of Izhikevich relative to HH: {:.2f}%'.format(perf_loss_izh_hh * 100))

# compute performance loss relative to Izhikevich with ThetaNeuron
perf_loss_izh_theta = (theta_acc - izh_acc) / theta_acc
print('Performance loss of Izhikevich relative to ThetaNeuron: {:.2f}%'.format(perf_loss_izh_theta * 100))

# compute performance loss of Izhikevich relative to HH
perf_loss_izh_srm = (srm_acc - izh_acc) / srm_acc
print('Performance loss of Izhikevich relative to SRM: {:.2f}%'.format(perf_loss_izh_srm * 100))

# compute performance loss of Izhikevich relative to IFSFA
perf_loss_izh_ifsfa = (ifsfa_acc - izh_acc) / ifsfa_acc
print('Performance loss of Izhikevich relative to IFSFA: {:.2f}%'.format(perf_loss_izh_ifsfa * 100))

# compute performance loss of Izhikevich relative to SRM
perf_loss_izh_qif = (qif_acc - izh_acc) / qif_acc
print('Performance loss of Izhikevich relative to QIF: {:.2f}%'.format(perf_loss_izh_qif * 100))



##################################################################### compute performance loss of SRM relative to LIF
perf_loss_srm_lif = (lif_acc - srm_acc) / lif_acc
print('Performance loss of SRM relative to LIF: {:.2f}%'.format(perf_loss_srm_lif * 100))

# compute performance loss of SRM relative to NLIF
perf_loss_srm_nlif = (nlif_acc - srm_acc) / nlif_acc
print('Performance loss of SRM relative to NLIF: {:.2f}%'.format(perf_loss_srm_nlif * 100))

# compute performance loss of SRM relative to AdEx
perf_loss_srm_adex = (adex_acc - srm_acc) / adex_acc
print('Performance loss of SRM relative to AdEx: {:.2f}%'.format(perf_loss_srm_adex * 100))

# compute performance loss of SRM relative to HH
perf_loss_srm_hh = (hh_acc - srm_acc) / hh_acc
print('Performance loss of SRM relative to HH: {:.2f}%'.format(perf_loss_srm_hh * 100))

# compute performance loss of SRM relative to Izhikevich
perf_loss_srm_izh = (izh_acc - srm_acc) / izh_acc
print('Performance loss of SRM relative to Izhikevich: {:.2f}%'.format(perf_loss_srm_izh * 100))

# compute performance loss relative to SRM with ThetaNeuron
perf_loss_srm_theta = (theta_acc - srm_acc) / theta_acc
print('Performance loss of SRM relative to ThetaNeuron: {:.2f}%'.format(perf_loss_srm_theta * 100))

# compute performance loss of SRM relative to IFSFA
perf_loss_srm_ifsfa = (ifsfa_acc - srm_acc) / ifsfa_acc
print('Performance loss of SRM relative to IFSFA: {:.2f}%'.format(perf_loss_srm_ifsfa * 100))

# compute performance loss of SRM relative to SRM
perf_loss_srm_qif = (qif_acc - srm_acc) / qif_acc
print('Performance loss of SRM relative to QIF: {:.2f}%'.format(perf_loss_srm_qif * 100))



############################################################### compute performance loss of IFSFA relative to LIF
perf_loss_ifsfa_lif = (lif_acc - ifsfa_acc) / lif_acc
print('Performance loss of IFSFA relative to LIF: {:.2f}%'.format(perf_loss_ifsfa_lif * 100))

# compute performance loss of IFSFA relative to NLIF
perf_loss_ifsfa_nlif = (nlif_acc - ifsfa_acc) / nlif_acc
print('Performance loss of IFSFA relative to NLIF: {:.2f}%'.format(perf_loss_ifsfa_nlif * 100))

# compute performance loss of IFSFA relative to AdEx
perf_loss_ifsfa_adex = (adex_acc - ifsfa_acc) / adex_acc
print('Performance loss of IFSFA relative to AdEx: {:.2f}%'.format(perf_loss_ifsfa_adex * 100))

# compute performance loss of IFSFA relative to HH
perf_loss_ifsfa_hh = (hh_acc - ifsfa_acc) / hh_acc
print('Performance loss of IFSFA relative to HH: {:.2f}%'.format(perf_loss_ifsfa_hh * 100))

# compute performance loss of IFSFA relative to Izhikevich
perf_loss_ifsfa_izh = (izh_acc - ifsfa_acc) / izh_acc
print('Performance loss of IFSFA relative to Izhikevich: {:.2f}%'.format(perf_loss_ifsfa_izh * 100))

# compute performance loss of IFSFA relative to SRM
perf_loss_ifsfa_srm = (srm_acc - ifsfa_acc) / srm_acc
print('Performance loss of IFSFA relative to SRM: {:.2f}%'.format(perf_loss_ifsfa_srm * 100))

# compute performance loss relative to IFSFA with ThetaNeuron
perf_loss_ifsfa_theta = (theta_acc - ifsfa_acc) / theta_acc
print('Performance loss of IFSFA relative to ThetaNeuron: {:.2f}%'.format(perf_loss_ifsfa_theta * 100))

# compute performance loss of IFSFA relative to QIF
perf_loss_ifsfa_qif = (qif_acc - ifsfa_acc) / qif_acc
print('Performance loss of IFSFA relative to QIF: {:.2f}%'.format(perf_loss_ifsfa_qif * 100))



######################################################################### compute performance loss of QIF relative to LIF
perf_loss_qif_lif = (lif_acc - qif_acc) / lif_acc
print('Performance loss of QIF relative to LIF: {:.2f}%'.format(perf_loss_qif_lif * 100))

# compute performance loss of QIF relative to NLIF
perf_loss_qif_nlif = (nlif_acc - qif_acc) / nlif_acc
print('Performance loss of QIF relative to NLIF: {:.2f}%'.format(perf_loss_qif_nlif * 100))

# compute performance loss of QIF relative to AdEx
perf_loss_qif_adex = (adex_acc - qif_acc) / adex_acc
print('Performance loss of QIF relative to AdEx: {:.2f}%'.format(perf_loss_qif_adex * 100))

# compute performance loss of QIF relative to HH
perf_loss_qif_hh = (hh_acc - qif_acc) / hh_acc
print('Performance loss of QIF relative to HH: {:.2f}%'.format(perf_loss_qif_hh * 100))

# compute performance loss of QIF relative to Izhikevich
perf_loss_qif_izh = (izh_acc - qif_acc) / izh_acc
print('Performance loss of QIF relative to Izhikevich: {:.2f}%'.format(perf_loss_qif_izh * 100))

# compute performance loss of QIF relative to SRM
perf_loss_qif_srm = (srm_acc - qif_acc) / srm_acc
print('Performance loss of QIF relative to SRM: {:.2f}%'.format(perf_loss_qif_srm * 100))

# compute performance loss of QIF relative to IFSFA
perf_loss_qif_ifsfa = (ifsfa_acc - qif_acc) / ifsfa_acc
print('Performance loss of QIF relative to IFSFA: {:.2f}%'.format(perf_loss_qif_ifsfa * 100))

# compute performance loss relative to QIF with ThetaNeuron
perf_loss_qif_theta = (theta_acc - qif_acc) / theta_acc
print('Performance loss of QIF relative to ThetaNeuron: {:.2f}%'.format(perf_loss_qif_theta * 100))








################################## compute performance loss of ThetaNeuron relative to LIF
perf_loss_theta_lif = (lif_acc - theta_acc) / lif_acc
print('Performance loss of ThetaNeuron relative to LIF: {:.2f}%'.format(perf_loss_theta_lif * 100))

# compute performance loss of ThetaNeuron relative to NLIF
perf_loss_theta_nlif = (nlif_acc - theta_acc) / nlif_acc
print('Performance loss of ThetaNeuron relative to NLIF: {:.2f}%'.format(perf_loss_theta_nlif * 100))

# compute performance loss of ThetaNeuron relative to AdEx
perf_loss_theta_adex = (adex_acc - theta_acc) / adex_acc
print('Performance loss of ThetaNeuron relative to AdEx: {:.2f}%'.format(perf_loss_theta_adex * 100))

# compute performance loss of ThetaNeuron relative to HH
perf_loss_theta_hh = (hh_acc - theta_acc) / hh_acc
print('Performance loss of ThetaNeuron relative to HH: {:.2f}%'.format(perf_loss_theta_hh * 100))

# compute performance loss of ThetaNeuron relative to Izhikevich
perf_loss_theta_izh = (izh_acc - theta_acc) / izh_acc
print('Performance loss of ThetaNeuron relative to Izhikevich: {:.2f}%'.format(perf_loss_theta_izh * 100))

# compute performance loss of ThetaNeuron relative to HH
perf_loss_theta_srm = (srm_acc - theta_acc) / srm_acc
print('Performance loss of ThetaNeuron relative to SRM: {:.2f}%'.format(perf_loss_theta_srm * 100))

# compute performance loss of ThetaNeuron relative to IFSFA
perf_loss_theta_ifsfa = (ifsfa_acc - theta_acc) / ifsfa_acc
print('Performance loss of ThetaNeuron relative to IFSFA: {:.2f}%'.format(perf_loss_theta_ifsfa * 100))

# compute performance loss of ThetaNeuron relative to SRM
perf_loss_theta_qif = (qif_acc - theta_acc) / qif_acc
print('Performance loss of ThetaNeuron relative to QIF: {:.2f}%'.format(perf_loss_theta_qif * 100))



