#ifndef CALCS_H_
#define CALCS_H_

#include <stdlib.h>
#include <stdio.h>
#include "mnist_load.h"
#include "nn_architecture.h"
#include "matrix.h"

// Macro to scale pixel values to range [0, 1]
#define PIXEL_SCALE(x) (((float) (x))/ 255.0)

// Function to initialize input activations with image pixels
void init_a(dataset *data, int offset, int input, int nb, float *a, float *a_d) {
    image *curr_image;
    for(int j = 0; j < nb; j++) {
        // Get current image
        curr_image = data->images[offset + j];

        // Scale pixel values and store in activations array
        for(int i = 0; i < input; i++) {
            a[i * nb + j] = PIXEL_SCALE(curr_image->pixels[i]);
        }
    }
    // Copy activations to device memory
    cudaMemcpy(a_d, a, nb * input * sizeof(float), cudaMemcpyHostToDevice);
}

// Function to perform forward propagation for a single layer
void forward_layer(nn *n, int l, float *inp_a, char func) {

    // Compute weighted sum of inputs and weights
    // z(l) = w(l) * a(l)
    multiply(n->layers[l]->z, n->layers[l]->weights, inp_a, n->layers[l]->out, n->layers[l]->in, n->layers[l]->size);

    // Add biases
    // z(l) += b(l)
    add(n->layers[l]->z, n->layers[l]->z, n->layers[l]->biases, n->layers[l]->out, n->layers[l]->size);

    // Apply activation function (softmax or ReLU)
    // a(l) = (softmax/relu)(z(l))
    if (func == 's') 
        softmax(n->layers[l]->a, n->layers[l]->z, n->layers[l]->out, n->layers[l]->size);
    else if (func == 'r')
        reLU(n->layers[l]->a, n->layers[l]->z, n->layers[l]->out, n->layers[l]->size);
}

// Function to perform forward propagation for the entire network
void forward_prop(nn *n, float *inp_a) {
    int nl  = n->num_layers;

    // If only 2 layers, forward through first layer with softmax
    if (nl == 2) {
        forward_layer(n, 1, inp_a, 's');
    } else {
        // Forward through first layer with ReLU
        forward_layer(n, 1, inp_a, 'r');

        // Forward through middle layers with ReLU
        for (int l = 2; l < nl - 1; l++) {
            forward_layer(n, l, n->layers[l-1]->a, 'r');
        }

        // Forward through last layer with softmax
        forward_layer(n, nl - 1, n->layers[nl-2]->a, 's');
    }
}

// Function to compute total cost using cross-entrpoy loss
void total_cost(dataset *data, int offset, nn *n, float *y_d) {
    int nl = n->num_layers;
    float *y = (float *)calloc(n->size * n->output_size, sizeof(float));
    for (int i = 0; i < n->size; i++) {
        int num = data->labels[offset + i];
        // One-hot encoding of labels
        for (int j = 0; j < n->output_size; j++) {
            if (j == num) y[j * n->size + i] = 1;
        }
    }

    // Copy labels to device memory
    cudaMemcpy(y_d, y, n->size * n->output_size * sizeof(float), cudaMemcpyHostToDevice);
    // Compute gradient of cost with respect to biases
    subtract(n->layers[nl-1]->grad_b, n->layers[nl-1]->a, y_d, n->output_size, n->size);
    free(y);
}

// Function to perform backpropagation for a single layer
void back_layer(float *out_del, float *inp_a, nn *n, int l, char func) {
    if (func == 'r') {
        // Compute derivative of ReLU
        // z(l) = relu'(z(l))
        reLU_prime(n->layers[l]->z, n->layers[l]->z, n->layers[l]->out, n->layers[l]->size);
        // Compute gradient of cost with respect to biases
        // grad_b(l) = out_del(l) . z(l)
        hadamard_product(n->layers[l]->grad_b, out_del, n->layers[l]->z, n->layers[l]->out, n->layers[l]->size);
    }

    // Compute gradient of cost with respect to weights
    // grad_w(l) = grad_b(l) * a(l-1)^T
    multiply_transpose(n->layers[l]->grad_w, n->layers[l]->grad_b, inp_a, n->layers[l]->out, n->size, n->layers[l]->in);
}

// Function to perform backpropagation for the entire network
void back_prop(nn *n, float *inp_a) {
    int nl = n->num_layers;

    // If only 2 layers, backpropagate through first layer with softmax
    if (nl == 2) {
        back_layer(n->layers[nl-1]->a, inp_a, n, nl-1, 's');
    }
    else {
        // Backpropagate through last layer with softmax
        back_layer(n->layers[nl-1]->a, n->layers[nl-2]->a, n, nl-1, 's');

        // Backpropagate through middle layers with ReLU
        for (int l = nl - 2; l > 1; l--) {
            transpose_multiply(n->layers[l]->a, n->layers[l+1]->weights, n->layers[l+1]->grad_b, n->layers[l]->out, n->layers[l+1]->out, n->layers[l]->size);
            back_layer(n->layers[l]->a, n->layers[l-1]->a, n, l, 'r');
        }

        // Backpropagate through first layer with ReLU
        transpose_multiply(n->layers[1]->a, n->layers[2]->weights, n->layers[2]->grad_b, n->layers[1]->out, n->layers[2]->out, n->layers[1]->size);
        back_layer(n->layers[1]->a, inp_a, n, 1, 'r');
    }
}   

// Function to train a single batch of data
void train_batch(dataset *data, int offset, nn *n, float *inp_a, float *a_d, float *y_d) {
    // Initialize activations with image pixels
    init_a(data, offset, n->input_size, n->size, inp_a, a_d);

    // Perform forward propagation
    forward_prop(n, a_d);

    // Compute total cost
    total_cost(data, offset, n, y_d);

    // Perform backpropagation
    back_prop(n, a_d);
}

// Function to update the weights and biases of the network using stochastic gradient descent
void update_nn(nn *n, float alpha) {
    float scalar = alpha / (float)n->size;

    // Loop over each layer in the network
    for(int l = 1; l < n->num_layers; l++) {
        // Scale the gradient of cost with respect to biases
        // grad_b(l) = scalar * grad_b(l)
        multiply_scalar(n->layers[l]->grad_b, n->layers[l]->grad_b, scalar, n->layers[l]->out, n->layers[l]->size);

        // Scale the gradient of cost with respect to weights
        // grad_w(l) = scalar * grad_w(l)
        multiply_scalar(n->layers[l]->grad_w, n->layers[l]->grad_w, scalar, n->layers[l]->out, n->layers[l]->in);

        // Update weights by subtracting the gradient of cost with respect to weights
        // w(l) = w(l) - grad_w(l)
        subtract(n->layers[l]->weights, n->layers[l]->weights, n->layers[l]->grad_w, n->layers[l]->out, n->layers[l]->in);

        // Update biases by subtracting the gradient of cost with respect to biases
        // b(l) = b(l) - grad_b(l)
        subtract_biases(n->layers[l]->biases, n->layers[l]->grad_b, n->layers[l]->out, n->layers[l]->size);
    }
}

#endif
