#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include "include/mnist_load.h"
#include "include/nn_architecture.h"
#include "include/calcs.h"
#include "include/matrix.h"

// Function to calculate the accuracy of the neural network
int calculate_accuracy(dataset *data, nn *network, int nb, float *inp_a, float *a_d, int size, int offset) {
    int correct = 0; // Counter for correct predictions
    int nl = network->num_layers; // Number of layers in the network

    // Allocate memory for the output of the network
    float *out_a = (float *)malloc(network->output_size * nb * sizeof(float));

    // Loop through the dataset in batches of nb
    for(int b = offset; b < offset+size; b += nb) {
        init_batch(network); // Initialize the batch
        init_a(data, b, network->input_size, nb, inp_a, a_d); // Initialize the input activations
        forward_prop(network, a_d); // Perform forward propagation
        cudaMemcpy(out_a, network->layers[nl-1]->a, network->output_size * nb  * sizeof(float), cudaMemcpyDeviceToHost); // Copy the output activations to host memory

        // Loop through the images in the batch
        for(int img = 0; img < nb; img++) {
            float max = out_a[0 + img]; // Maximum output activation
            int predict = 0; // Predicted class

            // Find the class with the highest output activation
            for (int n = 0; n < network->output_size; n++) {
                if(max < out_a[n * nb + img]) {
                    predict = n; 
                    max = out_a[n * nb + img];
                }
            }

            // If the predicted class matches the true class, increment the counter for correct predictions
            if(predict == data->labels[b + img]) { correct++; }
        }
    }

    free(out_a); // Free the allocated memory
    return correct; // Return the number of correct predictions
}

// Function to calculate the loss of the neural network
float calculate_loss(dataset *data, nn *network, int nb, float *inp_a, float *a_d, int size, int offset) {
    float loss = 0.0; // Total loss
    int nl = network->num_layers; // Number of layers in the network

    // Allocate memory for the output of the network
    float *out_a = (float *)malloc(network->output_size * nb * sizeof(float));

    // Loop through the dataset in batches of nb
    for (int b = offset; b < offset+size; b += network->size) {
        init_batch(network); // Initialize the batch
        init_a(data, b, network->input_size, nb, inp_a, a_d); // Initialize the input activations
        forward_prop(network, a_d); // Perform forward propagation
        cudaMemcpy(out_a, network->layers[nl-1]->a, network->output_size * nb  * sizeof(float), cudaMemcpyDeviceToHost); // Copy the output activations to host memory

        // Loop through the images in the batch
        for (int i = 0; i < nb; i++) {
            int y = data->labels[b + i]; // True class
            // Loop through the output neurons
            for (int j = 0; j < network->output_size; j++) {
                // If the output neuron corresponds to the true class, add the negative log of its activation to the loss
                loss += (j == y) ? -log(out_a[j * nb + i]) : 0.0;
            }
        }
    }

    free(out_a); // Free the allocated memory
    return loss/size; // Return the average loss
}

int main(int argc, char **argv) {
    // Check and parse command line arguments
    if (argc != 6) {
        printf("Usage %s <nl> <nh> <ne> <nb> <alpha>\n", argv[0]);
        printf("  nl: number of dense hidden layers\n");
        printf("  nh: number of neurons in each hidden layer\n");
        printf("  ne: number of epochs\n");
        printf("  nb: batch size\n");
        printf("  alpha: learning rate\n");
    }
    int nl = atoi(argv[1]); // Number of dense hidden layers
    int nh = atoi(argv[2]); // Number of neurons in each hidden layer
    int ne = atoi(argv[3]); // Number of epochs
    int nb = atoi(argv[4]); // Batch size
    float alpha = atof(argv[5]); // Learning rate

    printf("\nNeural Network activated with:\n");
    printf("nl: %d\n", nl);
    printf("nh: %d\n", nh);
    printf("ne: %d\n", ne);
    printf("nb: %d\n", nb);
    printf("alpha: %.3f\n", alpha);   
    printf("------------------------------------\n");

    // Initialize random number generator
    srand(time(NULL));

    // Open a file to write the loss
    FILE *fp = fopen("./Loss.txt" ,"a");

    // Initialize the clock
    clock_t start, end, total = 0;
    clock_t global_start = clock();

    // Define the neural network architecture
    int input_size = 784, output_size = 10, num_layers = nl + 2, layer_sizes[num_layers];
    layer_sizes[0] = input_size; 
    layer_sizes[nl + 1] = output_size; 
    for (int l = 1; l <= nl; l++) { layer_sizes[l] = nh; }

    // Create and initialize the neural network
    nn *network = create_nn(input_size, output_size, nb, num_layers, layer_sizes);
    init_nn(network);

    // Load the training and test datasets
    dataset *train_data = load_dataset("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    dataset *test_data = load_dataset("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    // Allocate memory for the input activations
    float *a = (float *)malloc(input_size * nb * sizeof(float));
    if (!a) printf("Failed to allocate memory for a\n");
    float *a_d, *y_d;
    assert(cudaMalloc((void **)&a_d, input_size * nb *sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void **)&y_d, output_size * nb *sizeof(float)) == cudaSuccess);
    
    float loss; int accuracy;
    int train_size = 50000;

    // Train the network for a number of epochs    
    for (int e = 0; e < ne; e++) {
        printf("Epoch: %d\n", e);
        start = clock();

        // Train the network for one epoch
        for (int b = 0; b < train_size; b += nb) {
            init_batch(network);
            train_batch(train_data, b, network, a, a_d, y_d);
            update_nn(network, alpha);
        }
        end = clock();
        total += end - start;
        
        // Calculate and print the validation loss and accuracy
        loss = calculate_loss(train_data, network, nb, a, a_d, 10000, 50000);
        accuracy = calculate_accuracy(train_data, network, nb, a, a_d, 10000, 50000);
        printf("Validation Loss: %f\n", loss);
        printf("Validation Accuracy: %d/10000: %.3f percent\n", accuracy, (float)accuracy/100);
        printf("-------------------------------------------\n");
        fprintf(fp, "%f\n", loss);
    }

    clock_t global_end = clock();
    printf("\n");

    // Calculate and print the test loss and accuracy
    loss = calculate_loss(test_data, network, nb, a, a_d, 10000, 0);
    accuracy = calculate_accuracy(test_data, network, nb, a, a_d, 10000, 0);
    printf("Test Loss: %f\n", loss);
    printf("Test Accuracy: %d/10000: %.3f percent\n", accuracy, (float)accuracy/100);

    printf("Training time: %.3f seconds\n", (float)total / CLOCKS_PER_SEC);
    printf("Total time: %.3f seconds\n", (float)(global_end - global_start) / CLOCKS_PER_SEC);
    printf("Grind Rate: %.3f images/second\n\n", ((float)ne * train_size)/((float)(total/CLOCKS_PER_SEC)));

    // Free allocated memory and close the file
    fclose(fp);
    free_dataset(train_data);
    free_dataset(test_data);
    free_nn(network);
    free(a);
    cudaFree(a_d); cudaFree(y_d);
}


