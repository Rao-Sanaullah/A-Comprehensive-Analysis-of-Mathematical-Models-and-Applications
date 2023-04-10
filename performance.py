import numpy as np
import matplotlib.pyplot as plt

# define LIF model
class LIF:
    def __init__(self, tau, v_reset, v_th, v_init=0.0, n_neurons=50):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.n_neurons = n_neurons
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))
        
    def update(self, I, dt):
        dvdt = (-self.v + I) / self.tau
        self.v += dvdt * dt
        if self.v >= self.v_th:
            spike = True
            self.v = self.v_reset
        else:
            spike = False
        return spike

# define NLIF model
class NLIF:
    def __init__(self, tau, v_reset, v_th, v_init=0.0, alpha=1.0, beta=1.0, n_neurons=50):
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.v = v_init
        self.alpha = alpha
        self.beta = beta
        self.n_neurons = n_neurons
        self.weights = np.random.normal(loc=0.0, size=(n_neurons, 1))
        
    def update(self, I, dt):
        dvdt = (-self.v + I) / self.tau
        self.v += dvdt * dt
        spike = self.v
        self.v = np.where(spike, self.v_reset + self.alpha * (self.v - self.v_th), self.v * self.beta)
        return spike


class AdEX:
    def __init__(self, tau_m, v_rheo, v_spike, delta_T, v_reset, v_init=0.0, n_neurons=50):
        self.tau_m = tau_m
        self.v_rheo = v_rheo
        self.v_spike = v_spike
        self.delta_T = delta_T
        self.v_reset = v_reset
        self.v = v_init
        self.n_neurons = n_neurons
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=(n_neurons, 1))
        
    def update(self, I, dt):
        dvdt = (-self.v + self.tau_m * I - self.v_rheo + self.delta_T * np.exp((self.v - self.v_spike) / self.delta_T)) / self.tau_m
        self.v += dvdt * dt
        spike = self.v >= self.v_spike
        self.v = np.where(spike, self.v_reset, self.v)
        return spike


######################################################## create dataset

n_samples = 1000
x1 = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
x2 = np.random.normal(loc=3.0, scale=1.0, size=n_samples)
X = np.hstack([x])
y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

# shuffle dataset
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
x = y[indices]

######################################################### train LIF model
lif = LIF(tau=4, v_reset=0.0, v_th=1.0, n_neurons=1000)
lif_spikes = np.zeros(len(X))
for i, x in enumerate(X):
    I = x
    spike = lif.update(I, dt=1.0)
    lif_spikes[i] = spike

########################################################## train NLIF model
nlif = NLIF(tau=4, v_reset=0.0, v_th=1.0, alpha=0.5, beta=0.5, n_neurons=1000)
nlif_spikes = np.zeros(len(y))
for i, x in enumerate(y):
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


##################################################### compute classification accuracy

lif_acc = np.mean(lif_spikes == x)
nlif_acc = np.mean(nlif_spikes == x)
adex_acc = np.mean(adex_spikes == x)

print('LIF accuracy: {:.2f}%'.format(lif_acc * 100))
print('NLIF accuracy: {:.2f}%'.format(nlif_acc * 100))
print('AdEX accuracy: {:.2f}%'.format(adex_acc * 100))

# compute performance loss of LIF relative to NLIF
perf_loss = (nlif_acc - lif_acc) / nlif_acc
print('Performance loss of LIF relative to NLIF: {:.2f}%'.format(perf_loss * 100))

# compute performance loss of AdEX relative to LIF
perf_loss_adex_lif = (lif_acc - adex_acc) / lif_acc
print('Performance loss relative to AdEX with LIF: {:.2f}%'.format(perf_loss_adex_lif * 100))

# compute performance loss relative to AdEX with NLIF
perf_loss_adex_nlif = (nlif_acc - adex_acc) / nlif_acc
print('Performance loss relative to AdEX with NLIF: {:.2f}%'.format(perf_loss_adex_nlif * 100))


#################################################################### plot results
fig, axs = plt.subplots(1, 3, figsize=(17, 4))
axs[0].scatter(X, y, c=lif_spikes, cmap='coolwarm', vmin=0, vmax=1)
axs[0].set_title('LIF')
axs[1].scatter(X, y, c=nlif_spikes, cmap='coolwarm', vmax=1)
axs[1].set_title('NLIF')
axs[2].scatter(X, y, c=adex_spikes, cmap='coolwarm', vmax=1)
axs[2].set_title('AdEX')

fig.suptitle('Classification Results')
#plt.show()

######################################################################### plot performance loss

fig, axs = plt.subplots(1, 3, figsize=(17, 4))

# plot 1
axs[0].plot(['LIF', 'NLIF'], [lif_acc, nlif_acc], '-o', color='blue', label='Accuracy')
axs[0].plot(['LIF', 'NLIF'], [1 - lif_acc, 1 - nlif_acc], '-o', color='orange', label='Error')
axs[0].set_ylim(0, 1)
axs[0].set_ylabel('Percentage')
axs[0].set_title('LIF vs. NLIF')
axs[0].text(0.5, 0.5, '{:.2f}%'.format(perf_loss * 100), ha='center', va='center', transform=axs[0, 1].transAxes, fontsize=14)
axs[0].legend()

# plot 2
axs[1].plot(['AdEX', 'LIF'], [adex_acc, lif_acc], '-o', color='blue', label='Accuracy')
axs[1].plot(['AdEX', 'LIF'], [1 - adex_acc, 1 - lif_acc ], '-o', color='orange', label='Error')
axs[1].set_ylim(0, 1)
axs[1].set_ylabel('Percentage')
axs[1].set_title('AdEX vs. LIF')
axs[1].text(0.5, 0.5, '{:.2f}%'.format(perf_loss_adex_lif * 100), ha='center', va='center', transform=axs[1].transAxes, fontsize=14)
axs[1].legend()

# plot 3
axs[2].plot(['AdEX', 'NLIF'], [adex_acc, nlif_acc], '-o', color='blue', label='Accuracy')
axs[2].plot(['AdEX', 'NLIF'], [1 - adex_acc, 1 - nlif_acc], '-o', color='orange', label='Error')
axs[2].set_ylim(0, 1)
axs[2].set_ylabel('Percentage')
axs[2].set_title('AdEX vs. NLIF')
axs[2].text(0.5, 0.5, '{:.2f}%'.format(perf_loss_adex_nlif * 100), ha='center', va='center', transform=axs[2].transAxes, fontsize=14)
axs[2].legend()


fig.suptitle('Performance Loss Results')
plt.show()