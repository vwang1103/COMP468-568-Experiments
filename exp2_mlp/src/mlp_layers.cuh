#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

struct LayerShape {
    int batch;
    int in_dim;
    int out_dim;
};

inline double layer_flops(const LayerShape& shape) {
    return 2.0 * static_cast<double>(shape.batch) * shape.in_dim * shape.out_dim;
}

inline double mlp_gflops(const std::vector<int>& layers, int batch, double millis) {
    double total_flops = 0.0;
    for (size_t i = 0; i + 1 < layers.size(); ++i) {
        LayerShape shape{batch, layers[i], layers[i + 1]};
        total_flops += layer_flops(shape);
    }
    return total_flops / (millis * 1e6);
}

// Bias add: activations[b,out] += bias[out]
// activations is row-major [batch][out_dim]
__global__ void bias_add_kernel(const float* __restrict__ bias,
                                float* __restrict__ activations,
                                LayerShape shape) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    if (idx >= elements) return;

    const int out_idx = static_cast<int>(idx % static_cast<size_t>(shape.out_dim));
    activations[idx] += bias[out_idx];
}

__global__ void relu_kernel(float* __restrict__ activations, size_t elements) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elements) return;

    const float x = activations[idx];
    activations[idx] = (x > 0.0f) ? x : 0.0f;
}

// tanh GELU approximation:
// 0.5*x*(1+tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
__device__ __forceinline__ float gelu_approx(float x) {
    const float kSqrt2OverPi = 0.7978845608028654f;
    const float x3 = x * x * x;
    const float u = kSqrt2OverPi * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(u));
}

__global__ void gelu_kernel(float* __restrict__ activations, size_t elements) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elements) return;

    activations[idx] = gelu_approx(activations[idx]);
}

inline void launch_bias_add(const float* bias, float* activations, const LayerShape& shape, cudaStream_t stream) {
    const int threads = 256;
    const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    bias_add_kernel<<<blocks, threads, 0, stream>>>(bias, activations, shape);
}

inline void launch_activation(const std::string& activation,
                              float* activations,
                              const LayerShape& shape,
                              cudaStream_t stream) {
    const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    if (activation == "relu") {
        relu_kernel<<<blocks, threads, 0, stream>>>(activations, elements);
    } else if (activation == "gelu") {
        gelu_kernel<<<blocks, threads, 0, stream>>>(activations, elements);
    } else {
        // unknown activation => do nothing
    }
}

// Fused bias + activation
// activation_type: 0=relu, 1=gelu
__global__ void fused_bias_activation_kernel(const float* __restrict__ bias,
                                             float* __restrict__ activations,
                                             LayerShape shape,
                                             int activation_type) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    if (idx >= elements) return;

    const int out_idx = static_cast<int>(idx % static_cast<size_t>(shape.out_dim));
    float x = activations[idx] + bias[out_idx];

    if (activation_type == 1) x = gelu_approx(x);
    else x = (x > 0.0f) ? x : 0.0f;

    activations[idx] = x;
}

inline void launch_fused_bias_activation(const float* bias,
                                         const std::string& activation,
                                         float* activations,
                                         const LayerShape& shape,
                                         cudaStream_t stream) {
    const int activation_type = (activation == "gelu") ? 1 : 0;
    const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    fused_bias_activation_kernel<<<blocks, threads, 0, stream>>>(bias, activations, shape, activation_type);
}

inline void run_gemm_layer(const float* input,
                           const float* weight,
                           float* output,
                           const LayerShape& shape,
                           cublasHandle_t handle) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    const int m = shape.out_dim;  // rows of C_cm (out x batch)
    const int n = shape.batch;    // cols of C_cm
    const int k = shape.in_dim;   // shared dim

    const int lda = k;            // rows of W_cm (in x out)
    const int ldb = k;            // rows of A_cm (in x batch)
    const int ldc = m;            // rows of C_cm (out x batch)

    cublasStatus_t st = cublasSgemm(handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    m, n, k,
                                    &alpha,
                                    weight, lda,
                                    input,  ldb,
                                    &beta,
                                    output, ldc);
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSgemm failed in run_gemm_layer");
    }
}