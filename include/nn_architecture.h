#ifndef NN_ARCHITECTURE_H_
#   define NN_ARCHITECTURE_H_

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>
#include "mnist_load.h"

// Define the structure of a layer in the neural network
typedef struct layer_ {
    int in, out, size; // Input size, output size, and batch size
    float *weights, *biases; // Weights and biases
    float *a, *z, *grad_b, *grad_w; // Activations, pre-activations, gradients of biases and weights
} layer;

// Define the structure of the neural network
typedef struct nn_ {
    int input_size, output_size, size; // Input size, output size, and batch size
    int num_layers; // Number of layers
    int *layer_sizes; // Sizes of each layer
    layer **layers; // Layers of the network
} nn;

// Function to create a layer with given input size, output size, and batch size
layer * create_layer(int in, int out, int size) {
    layer *l = (layer *)malloc(sizeof(layer));
    if(!l) printf("Failed to allocate memory for layer\n");

    l->in = in;
    l->out = out;
    l->size = size;

    assert(cudaMalloc((void **)&(l->weights), out * in *sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void **)&(l->biases), out * size *sizeof(float)) == cudaSuccess);

    assert(cudaMalloc((void **)&(l->a), out * size *sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void **)&(l->z), out * size *sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void **)&(l->grad_b), out * size *sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void **)&(l->grad_w), out * in *sizeof(float)) == cudaSuccess);

    return l;
}

// Function to create a neural network with given input size, output size, batch size, number of layers, and sizes of each layer
nn * create_nn(int input, int output, int size, int num_layers, int *layer_sizes) {
    nn *network = (nn *)malloc(sizeof(nn));
    if(!network) printf("Failed to allocate memory for network\n");

    network->input_size = input;
    network->output_size = output;
    network->size = size;
    network->num_layers = num_layers;
    network->layer_sizes = layer_sizes;

    network->layers = (layer **)malloc(num_layers * sizeof(layer *));
    if(!network->layers) printf("Failed to allocate memory for layers\n");

    network->layers[0] = NULL;
    for(int l = 1; l < num_layers; l++) {
        network->layers[l] = create_layer(layer_sizes[l-1], layer_sizes[l], size);
    }

    return network;
}

// Function to initialize a layer with weights according to Kaiming-He initialization scheme and zero biases
void init_layer(layer *l, int num) {
    float *weights = (float *)malloc(l->out * l->in * sizeof(float));
    for (int i = 0; i < l->out * l->in; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        weights[i] = sqrt( -2 * log(u1) ) * cos(2 * 3.14159 * u2) * (2 / (float)num);
    }
    cudaMemcpy(l->weights, weights, l->out * l->in * sizeof(float), cudaMemcpyHostToDevice);
    float *biases = (float *)malloc(l->out * l->size * sizeof(float));
    for (int i = 0; i < l->out * l->size; i++) {
        biases[i] = 0.0;
    }
    cudaMemcpy(l->biases, biases, l->out * l->in * sizeof(float), cudaMemcpyHostToDevice);
    free(weights); free(biases);
}

void init_nn(nn *network) {
    for (int l = 1; l < network->num_layers; l++) {
        init_layer(network->layers[l], network->layer_sizes[l-1]);
    }
}

// Function to initialize gradients of a layer with zeroes
void init_grads(layer *l) {
    int max = (l->size > l->in) ? l->size : l->in; 
    float *zeroes = (float *)malloc(l->out * max * sizeof(float));
    for (int i = 0; i < l->out * max; i++) {
        zeroes[i] = 0.0;
    }
    cudaMemcpy(l->grad_w, zeroes, l->out * l->in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l->grad_b, zeroes, l->out * l->size * sizeof(float), cudaMemcpyHostToDevice);
    free(zeroes);
}

// Function to set gradient values to zero at the start of a batch 
void init_batch(nn *network) {
    for (int l = 1; l < network->num_layers; l++) {
        init_grads(network->layers[l]);
    }
}

// Function to free a layer
void free_layer(layer *l) {
    cudaFree(l->weights);
    cudaFree(l->biases);
    cudaFree(l->a);
    cudaFree(l->z);
    cudaFree(l->grad_b);
    cudaFree(l->grad_w);
    free(l);
}

// Function to free a neural network
void free_nn(nn *network) {
    for(int l = 1; l < network->num_layers; l++) {
        free_layer(network->layers[l]);
    }
    free(network->layers);
    free(network);
}

#endif
