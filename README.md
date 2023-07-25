# A Comprehensive Analysis of Mathematical Models and Applications


This repository discusses Spiking Neural Networks (SNNs) and their mathematical models, which simulate the behavior of neurons through the generation of spikes [1]. These models are used to improve the accuracy of the network's outputs and have potential applications in various fields. However, implementing SNNs can pose certain challenges, especially when it comes to determining the most suitable SNN model for classification tasks requiring high accuracy and low-performance loss. To address this issue, a study was conducted to compare the performance, behavior, and spike generation of different SNN models using the same inputs and neurons. The results of the study provide valuable insights for researchers and practitioners in this area, highlighting the importance of comparing different models to determine the most effective one.


# Performance Comparison for Classification Tasks

In this study, we investigate the behavior of various SNN models through simulations based on their respective equations. The models were implemented using an update method that determines whether a spike occurred or not based on the input current and time step. We evaluated the performance of each model by measuring its classification accuracy and performance loss. We also randomly initialized weights for each model and visualized the spiking activity of neurons over time. Figure \ref{per} demonstrates the performance loss between different models. We executed each model using 1000 samples as inputs and 1000 neurons, although other parameters varied for each model.


# Dataset

Datasets play a crucial role in the development and evaluation of machine learning models, including spiking neural networks. The synthetic dataset used in this study was designed to have two classes that are easily separable by a linear classifier, which allows for a straightforward evaluation of the performance of different models. Additionally, the synthetic dataset is more transparent in terms of the underlying data generation process, which can help in identifying the strengths and weaknesses of different models.

# References:

[1] Samanwoy Ghosh-Dastidar and Hojjat Adeli. Spiking neural networks. 2009.

[2] Doron Tal and Eric L Schwartz. Computing with the leaky integrate-and-fire neuron: logarithmic computation and multiplication. 1997.

[3] Wulfram Gerstner and Romain Brette. Adaptive exponential integrate-and-fire model. Scholarpedia, 2009.

[4] Renaud Jolivet, Timothy J Lewis, and Wulfram Gerstner. Generalized integrateand-fire models of neuronal activity approximate spike trains of a detailed model to a high degree of accuracy. Journal of neurophysiology, 92(2):959–976, 2004.



For any help, please contact

Sanaullah (sanaullah@fh-bielefeld.de)
