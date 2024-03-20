#ifndef MNIST_LOAD_H_
#define MNIST_LOAD_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <zlib.h>

// Define the size of the training and validation sets
#define TRAIN 50000
#define VALIDATION 10000

// Structure to represent a single MNIST image
typedef struct mnist_image_ {
    uint8_t pixels[784]; // Structure to represent a dataset of MNIST images
} image;

// Structure to represent a dataset of MNIST images
typedef struct mnist_dataset_ {
    image **images; // Pointer to an array of image pointers
    uint8_t *labels; // Pointer to an array of labels
    uint32_t size; // Size of the dataset
} dataset;

dataset *load_dataset(const char *image_path, const char *label_path);
void free_dataset(dataset *data);

// Function to swap the endianness of a 32-bit integer
uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val << 8)  & 0x00ff0000) |
           ((val >> 8)  & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

// Function to load labels from a MNIST label file
uint8_t * load_labels(const char *path, uint32_t *num_labels) {
    uint8_t *labels;
    uint32_t magic, num_items;

    // Open the file
    gzFile file = gzopen(path, "rb");
    if (NULL == file) {
        fprintf(stderr, "Could not open file: %s\n", path);
        return NULL;
    }
    // Read the magic number and the number of items
    gzread(file, &magic, sizeof(magic));
    if (swap_endian(magic) != 2049) {
        printf("Error: Invalid MNIST label file %s\n", path);
        return NULL;
    }
    gzread(file, &num_items, sizeof(num_items));
    num_items = swap_endian(num_items);

    // Read the labels
    labels = (uint8_t*)malloc(num_items * sizeof(uint8_t));
    for (int i = 0; i < num_items; i++) {
        uint8_t label;
        gzread(file, &label, sizeof(label));
        labels[i] = label;
    }

    gzclose(file);
    *num_labels = num_items;
    return labels;
}

// Function to load images from a MNIST image file
image ** load_images(const char *path, uint32_t *num_images) {
    image **images;
    uint32_t magic, num_items, num_rows, num_cols;

    // Open the file
    gzFile file = gzopen(path, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", path);
        return NULL;
    }
    // Read the magic number and the number of items and the size of the images
    gzread(file, &magic, sizeof(magic));
    magic = swap_endian(magic);
    if (magic != 2051) {
        printf("Error: Invalid MNIST image file %s\n", path);
        return NULL;
    }
    gzread(file, &num_items, sizeof(num_items));
    num_items = swap_endian(num_items);
    gzread(file, &num_rows, sizeof(num_rows));
    num_rows = swap_endian(num_rows);
    gzread(file, &num_cols, sizeof(num_cols));
    num_cols = swap_endian(num_cols);

    // Read the images
    images = (image **)malloc(sizeof(image *) * num_items);
    for (int i = 0; i < num_items; i++) {
        images[i] = (image *)malloc(sizeof(image));
        for (int r = 0; r < num_rows; r++) {
            for (int c = 0; c < num_cols; c++) {
                gzread(file, &images[i]->pixels[r * num_cols + c], sizeof(uint8_t));
            }
        }
    }

    gzclose(file);
    *num_images = num_items;
    return images;
}

// Function to load a dataset from a pair of MNIST files
dataset * load_dataset(const char *image_path, const char *label_path) {
    dataset *data;
    uint32_t num_imgs, num_lbls;

    data = (dataset *)malloc(sizeof(dataset));
    data->images = load_images(image_path, &num_imgs);
    data->labels = load_labels(label_path, &num_lbls);
    data->size = num_imgs;
    return data;
}

// Function to free a dataset
void free_dataset(dataset *data) {
    for (int i = 0; i < data->size; i++) {
        free(data->images[i]);
    }
    free(data->images);
    free(data->labels);
    free(data);
}

#endif
