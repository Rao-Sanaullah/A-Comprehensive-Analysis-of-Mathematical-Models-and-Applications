# Evaluating-Spiking-Neural-Network-Models-A-Comparative-Performance-Analysis


This repository discusses Spiking Neural Networks (SNNs) and their mathematical models, which simulate the behavior of neurons through the generation of spikes [1]. These models are used to improve the accuracy of the network's outputs and have potential applications in various fields. However, implementing SNNs can pose certain challenges, especially when it comes to determining the most suitable SNN model for classification tasks requiring high accuracy and low performance loss. To address this issue, a study was conducted to compare the performance, behavior, and spike generation of different SNN models using the same inputs and neurons. The results of the study provide valuable insights for researchers and practitioners in this area, highlighting the importance of comparing different models to determine the most effective one.


# Performance Comparison for Classification Tasks

In this study, we investigate the behavior of various SNN models through simulations based on their respective equations. The models were implemented using an update method that determines whether a spike occurred or not based on the input current and time step. We evaluated the performance of each model by measuring its classification accuracy and performance loss. We also randomly initialized weights for each model and visualized the spiking activity of neurons over time. Figure \ref{per} demonstrates the performance loss between different models. We executed each model using 1000 samples as inputs and 1000 neurons, although other parameters varied for each model.


  1- Classification Results
  
![Classification Results](https://github.com/Rao-Sanaullah/Evaluating-Spiking-Neural-Network-Models-A-Comparative-Performance-Analysis/blob/main/2.png)

  2- Performance Loss
  
![Performance Loss](https://github.com/Rao-Sanaullah/Evaluating-Spiking-Neural-Network-Models-A-Comparative-Performance-Analysis/blob/main/1.png)


The performance loss between LIF vs NLIF, AdEX vs LIF, and AdEX vs NLIF model. The performance of each two compared models is measured in terms of their accuracy. We used 1000 nsamples as inputs and 1000 neurons for each model execution. However, the values of other parameters varied for each model.


These results show the classification accuracy and performance loss of different SNN models. In other words, LIF model had an accuracy of 71.65%, while NLIF had an accuracy of 67.05%. AdEX model had the highest accuracy of 90.65%. The performance loss of LIF model was -6.86% relative to NLIF, -26.52% relative to AdEX with LIF, and -35.20% relative to AdEX with NLIF. These results provide insights into the suitability of different SNN models for classification tasks, and can aid in selecting the appropriate model for a given task.


# Membrane potential (mV) / Spikes

The simulation of different neuron models: the leaky integrate-and-fire (LIF) model [2], the nonlinear LIF (NLIF) model [3], and the adaptive exponential (AdEx) model [4]. Each neuron model is defined by a set of differential equations that describe how the neuron's voltage and other state variables change over time in response to input currents.


- Voltage trace and Spike raster plots

  1- LIF neuron model
  
![Classification Results](https://github.com/Rao-Sanaullah/Evaluating-Spiking-Neural-Network-Models-A-Comparative-Performance-Analysis/blob/main/s1.png)

  2- NLIF neuron model
  
![Performance Loss](https://github.com/Rao-Sanaullah/Evaluating-Spiking-Neural-Network-Models-A-Comparative-Performance-Analysis/blob/main/s2.png)

  3- AdEX neuron model

![Performance Loss](https://github.com/Rao-Sanaullah/Evaluating-Spiking-Neural-Network-Models-A-Comparative-Performance-Analysis/blob/main/s3.png)

# Dataset

The dataset for this study was generated using the following approach; Let $n_{samples} = 1000$, $x_1 \sim \mathcal{N}(0,1)$, $x_2 \sim \mathcal{N}(3,1)$, $X = \begin{bmatrix} x_1 & x_2 \end{bmatrix}$, $y = \begin{bmatrix} 0_{n_{samples}} & 1_{n_{samples}} \end{bmatrix}$, where $0_{n_{samples}}$ and $1_{n_{samples}}$ are the vectors of length $n_{samples}$ filled with zeros and ones, respectively. To shuffle the dataset, let $indices = \begin{bmatrix} 0 & 1 & \cdots & 2n_{samples}-1 \end{bmatrix}$, and apply a random permutation to $indices$. Then, let $X$ and $y$ be the arrays obtained by indexing $X$ and $y$ with the shuffled $indices$.

where $n = 1000$ is the number of samples, $x_{1,i} \sim \mathcal{N}(0,1)$ and $x_{2,i} \sim \mathcal{N}(3,1)$ are the features of the $i$-th sample for $i \in {1, \dots, n}$, and $y$ is a vector of length $2n$ where the first $n$ elements are 0 and the last $n$ elements are 1. The dataset is then shuffled using the indices $indices = [0, 1, \dots, 2n-1]$, and $X$ and $y$ are updated accordingly.

# References:

[1] Samanwoy Ghosh-Dastidar and Hojjat Adeli. Spiking neural networks. 2009.

[2] Doron Tal and Eric L Schwartz. Computing with the leaky integrate-and-fire neuron: logarithmic computation and multiplication. 1997.

[3] Wulfram Gerstner and Romain Brette. Adaptive exponential integrate-and-fire model. Scholarpedia, 2009.

[4] Renaud Jolivet, Timothy J Lewis, and Wulfram Gerstner. Generalized integrateand-fire models of neuronal activity approximate spike trains of a detailed model to a high degree of accuracy. Journal of neurophysiology, 92(2):959–976, 2004.



For any help, please contact

Sanaullah (sanaullah@fh-bielefeld.de)
