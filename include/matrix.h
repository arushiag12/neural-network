#ifndef MATRIX_H_
#   define MATRIX_H_

#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <assert.h>

int NTPB = 128;

// C(n,m) = A(n,m) + B(n,m)
__global__ void add_kernel(float *C, float *A, float *B, int n, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n * m) {
        C[tid] = A[tid] + B[tid];
    }
}
void add(float *C, float *A, float *B, int n, int m) {
    int ntpb = NTPB, numblocks = (n * m / ntpb) + 1;
    add_kernel<<< numblocks, ntpb >>>(C, A, B, n, m);
    cudaDeviceSynchronize();
}

// C(n,m) = A(n,m) - B(n,m)
__global__ void subtract_kernel(float *C, float *A, float *B, int n, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n * m) {
        C[tid] = A[tid] - B[tid];
    }
}
void subtract(float *C, float *A, float *B, int n, int m) {
    int ntpb = NTPB, numblocks = (n * m / ntpb) + 1;
    subtract_kernel<<< numblocks, ntpb >>>(C, A, B, n, m);
    cudaDeviceSynchronize();
}

// C(n,m) = A(n,p) * B(p,m)
void multiply(float *C, float *A, float *B, int n, int p, int m) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float one = 1, zero = 0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, p, &one, B, m, A, p, &zero, C, m);
    cublasDestroy(handle);
}

// C(n,m) = A(n,m) . B(n,m)
__global__ void hadamard_product_kernel(float *C, float *A, float *B, int n, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n * m) {
        C[tid] = A[tid] * B[tid];
    }
}
void hadamard_product(float *C, float *A, float *B, int n, int m) {
    int ntpb = NTPB, numblocks = (n * m / ntpb) + 1;
    hadamard_product_kernel<<< numblocks, ntpb >>>(C, A, B, n, m);
    cudaDeviceSynchronize();
}

// C(n,m) = A(n,p) * B(m,p)^T
void multiply_transpose(float *C, float *A, float *B, int n, int p, int m) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float one = 1, zero = 0;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, p, &one, B, p, A, p, &zero, C, m);
    cublasDestroy(handle);
}

// C(n,m) = A(p,n)^T * B(p,m)
void transpose_multiply(float *C, float *A, float *B, int n, int p, int m) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float one = 1, zero = 0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, p, &one, B, m, A, n, &zero, C, m);
    cublasDestroy(handle);
}

// C(n,m) = scalar * A(n,m)
__global__ void multiply_scalar_kernel(float *C, float *A, float scalar, int n, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n * m) {
        C[tid] = A[tid] * scalar;
    }
}
void multiply_scalar(float *C, float *A, float scalar, int n, int m) {
    int ntpb = NTPB, numblocks = (n * m / ntpb) + 1;
    multiply_scalar_kernel<<< numblocks, ntpb >>>(C, A, scalar, n, m);
    cudaDeviceSynchronize();
}

// Subtracts delta of biases from biases
__global__ void subtract_biases_kernel(float *C, float *A, int n, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float sum = 0;
        for (int i = 0; i < m; i++) {
            sum += A[tid * m + i];
        }
        for (int i = 0; i < m; i++) {
            C[tid * m + i] -= sum;
        }
    }
}
void subtract_biases(float *C, float *A, int n, int m) {
    int ntpb = NTPB, numblocks = (n / ntpb) + 1;
    subtract_biases_kernel<<< numblocks, ntpb >>>(C, A, n, m);
    cudaDeviceSynchronize();
}

// C(n,m) = reLU(A(n,m))
__global__ void reLU_kernel(float *C, float *A, int n, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n * m) {
        C[tid] = (A[tid] > 0) ? A[tid] : 0.0; 
    }
}
void reLU(float *C, float *A, int n, int m) {
    int ntpb = NTPB, numblocks = (n * m / ntpb) + 1;
    reLU_kernel<<< numblocks, ntpb >>>(C, A, n, m);
    cudaDeviceSynchronize();
}

// C(n,m) = softmax(A(n,m))
__global__ void softmax_kernel(float *C, float *A, int n, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < m) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += exp(A[i * m + tid]);
        }
        for (int i = 0; i < n; i++) {
            C[i * m + tid] = exp(A[i * m + tid]) / sum;
        }
    }
}
void softmax(float *C, float *A, int n, int m) {
    int ntpb = NTPB, numblocks = (m / ntpb) + 1;
    softmax_kernel<<< numblocks, ntpb >>>(C, A, n, m);
    cudaDeviceSynchronize();
}

// C(n,m) = reLU_prime(A(n,m))
__global__ void reLU_prime_kernel(float *C, float *A, int n, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n * m) {
        C[tid] = (A[tid] > 0) ? 1.0 : 0.0; 
    }
}
void reLU_prime(float *C, float *A, int n, int m) {
    int ntpb = NTPB, numblocks = (n * m / ntpb) + 1;
    reLU_prime_kernel<<< numblocks, ntpb >>>(C, A, n, m);
    cudaDeviceSynchronize();
}

#endif
