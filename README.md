#  Neural Network in C using CUDA and CUBLAS

## Overview

This project focuses on implementing a neural network in C with CUDA and cuBLAS acceleration for training on the MNIST database. The neural network's architecture is customizable, taking as input various parameters such as the number of hidden layers, neurons in each layer, epochs for training, batch size, and learning rate.

The neural network architecture utilizes the rectified linear unit (ReLU) activation function for all hidden layers and employs softmax activation for the output layer. ReLU is chosen for its simplicity and effectiveness in handling non-linearities in the data, while softmax is ideal for multi-class classification tasks like recognizing handwritten digits in the MNIST dataset.

The project takes advantage of CUDA and cuBLAS libraries for accelerated matrix-matrix operations, significantly speeding up the training process. This efficient parallelization enables the neural network to process large amounts of data and perform complex computations much faster than traditional CPU-based implementations.

## Key Features

- Neural network for training on the MNIST database.
- Customizable architecture.
- Utilizes ReLU activation for hidden layers and softmax for the output layer.
- Accelerated matrix-matrix products using CUDA and cuBLAS for efficient training.

## Instructions

### Prerequisites

Load the required modules:
```bash
$ module load cuda
$ module load openblas
```

### Compilation

To compile the project, use the following command:
```bash
$ make
```

### Running the Code

- `<nl>`: Number of hidden layers
- `<nh>`: Number of neurons in each hidden layer
- `<ne>`: Number of epochs
- `<nb>`: Batch size
- `<alpha>`: Learning rate

Execute the code with:
```bash
$ ./nn.out <nl> <nh> <ne> <nb> <alpha>
```

### Cleanup

Run the following to cleanup:
```bash
$ make clean
```
