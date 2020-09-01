# Day 4
## MLFlow
It's an open source framework for operational flow of ML models. It includes experimentation, model development and scalable deployment in the cloud. It it an extensive framework and is used by major corporations.

`pip install mlflow`

## Graffitist
It's an approach towards quantization of ML models without much decrease in accuracy. It uses the technique of Training Quantization thresholds (TQT) using standard backpropagation and gradient descent.
> It has reached max ~70% accuracy on ImageNet data.

## Quantum Computing and Neural Networks
Quantum computers (QC) have dynamic bits called qubits (super-position of 0 and 1) contrary to traditional computing with static bits (0 and 1). This increases the computational power manifold and is the reason for the computational supremacy of QCs
The biggest QC made is of 53 qubits by Google and IBM separately. 
> The QCs are currently not more than research fields because of the noise and errors encountered in the hardware due to entanglement and superposition.

## Graph Neural Network (GNNs)
A graph is a representation of edges and vertices which may be uni or bi directional, G = (E, V). It has better accuracy as there can be millions of connections between the nodes and the graph's efficiency improves over time. This data in the graph is not the regular structured data in tensors, which makes it more powerful while establishing patterns. An initial partially labeled graph will learn to predict all the non-labelled nodes by analysing the patterns in the labelled nodes. This prediction accuracy steeply increases with time.
### Popular Open-source GNN libraries
1. Pytorch Geometric
2. Graph Nets; by Google's DeepMind; made for use on top of tensorflow
3. DeepGraph; useful for beginners
### Geometric Deep Learning
Shift from regular deep learning using Euclidian datasets (flat planes) to 3D object datasets like graphs, Point Clouds, etc.
**Graph Convolutional Networks (GCNs)** use graph inputs and use representations from them which are then layered (like regular ConvNets) and the output graph obtained already has the inherent representations and relations even before the training process.




