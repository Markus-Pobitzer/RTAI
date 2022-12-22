# Reliable and Trustworthy Artificial Intelligence 2022 Course Project
Mandatory project of the [Reliable and Trustworthy Artificial Intelligence](https://www.sri.inf.ethz.ch/teaching/rtai22)
(RTAI) course at ETH. Thought by [Prof. Dr. Martin Vechev](https://www.sri.inf.ethz.ch/people/martin).


The template code for the project can be downloaded here: [link](https://files.sri.inf.ethz.ch/website/teaching/reliableai2022/materials/project/project_release.zip).

## Task

This project implements a DeepPoly verifier for fully connected, convolutional, and residual neural networks.
The verifier takes as input an image from the MNIST or CIFAR-10 dataset, and an epsilon value, and tests the image
within an epsilon-ball. The output is either "verified" or "not verified", depending on whether the verifier can prove
that the network will always output the true label for the given input and epsilon.

## 𝜶-learning
To improve the ReLU transformer in DeepPoly, you can follow these steps:

1) Identify the neurons in the network that you want to optimize the ReLU transformer for. In our case this includes all
   neurons in the network.

2) For each neuron, define a optimization problem that aims to find the value of 𝜶 that maximizes the precision of the
   verifier. This optimization problem involves minimizing a loss function that represents the error between the predicted and true outputs of the network.

3) Solve the optimization problem for each neuron using an optimization algorithm such as gradient descent. 
   This will allow you to learn the value of 𝜶 for each neuron that maximizes the precision of the verifier.

4) Once you have learned the values of 𝜶 for all of the neurons, use these values to construct a new ReLU transformer for each neuron.

5) Run the verification procedure using the new ReLU transformers to test the precision of the verifier.

6) Repeat until we have verified the network or we time out.

## Some tipps

- The 𝜶 values have to be between [0, 1]. To bring them to the given range use an activation function such as the Sigmoid function.
- In the loss we compute something like loss = lb[true_label] - ub, where lb is the lower bound and ub the upper bound.
  Put this as a separate layer in your Verifier such that you can make a forward pass and also backsubstitude through it.
   See our FinalLossVerifier class.

## DeepPoly
DeepPoly is a tool for verifying the robustness and correctness of deep neural networks (DNNs). It is based on the
DeepPoly framework, which is a technique for formally verifying properties of DNNs. The DeepPoly framework works by
constructing a mathematical model of the DNN and the input data, and then checking whether the model satisfies a given
property.

DeepPoly is particularly useful for verifying the robustness of DNNs, as it can help identify inputs for which the DNN
may produce incorrect outputs, and can provide guarantees about the robustness of the DNN under certain types of
perturbations. It can also be used to verify the correctness of DNNs, by checking that the DNN satisfies certain desired
properties, such as the ability to classify inputs correctly or to predict certain outcomes.


## Credits

This project was developed by Markus Pobitzer and Nadine Frischknecht.
