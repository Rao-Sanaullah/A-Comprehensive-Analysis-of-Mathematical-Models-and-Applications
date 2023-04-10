# Evaluating-Spiking-Neural-Network-Models-A-Comparative-Performance-Analysis


This repository discusses Spiking Neural Networks (SNNs) and their mathematical models, which simulate the behavior of neurons through the generation of spikes. These models are used to improve the accuracy of the network's outputs and have potential applications in various fields. However, implementing SNNs can pose certain challenges, especially when it comes to determining the most suitable SNN model for classification tasks requiring high accuracy and low performance loss. To address this issue, a study was conducted to compare the performance, behavior, and spike generation of different SNN models using the same inputs and neurons. The results of the study provide valuable insights for researchers and practitioners in this area, highlighting the importance of comparing different models to determine the most effective one.


# Performance Comparison for Classification Tasks

In this study, we investigate the behavior of various SNN models through simulations based on their respective equations. The models were implemented using an update method that determines whether a spike occurred or not based on the input current and time step. We evaluated the performance of each model by measuring its classification accuracy and performance loss. We also randomly initialized weights for each model and visualized the spiking activity of neurons over time. Figure \ref{per} demonstrates the performance loss between different models. We executed each model using 1000 samples as inputs and 1000 neurons, although other parameters varied for each model.


![Classification Results](https://github.com/Rao-Sanaullah/Evaluating-Spiking-Neural-Network-Models-A-Comparative-Performance-Analysis/blob/main/2.png)
