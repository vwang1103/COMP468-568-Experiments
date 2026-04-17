#pragma once

#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>

#include <array>
#include <string>
#include <vector>

struct LenetShape {
    int batch;
    // Assume MNIST-style 1x32x32 input.
    static constexpr int in_channels = 1;
    static constexpr int in_height = 32;
    static constexpr int in_width = 32;

    static constexpr int conv1_out_channels = 6;
    static constexpr int conv1_kernel = 5;
    static constexpr int conv2_out_channels = 16;
    static constexpr int conv2_kernel = 5;

    static constexpr int pool_stride = 2;

    static constexpr int fc1_out = 120;
    static constexpr int fc2_out = 84;
    static constexpr int fc3_out = 10;

    size_t input_elements;
    size_t conv1_out_elems;
    size_t pool1_out_elems;
    size_t conv2_out_elems;
    size_t pool2_out_elems;
    size_t fc1_out_elems;
    size_t fc2_out_elems;
    size_t output_elements;

    size_t total_weight_elements;
    size_t total_bias_elements;
    std::vector<size_t> weight_offsets;
    std::vector<size_t> bias_offsets;
};

inline LenetShape make_lenet_shape(int batch) {
    LenetShape s{};
    s.batch = batch;
    const int conv1_out_h = LenetShape::in_height - LenetShape::conv1_kernel + 1;  // stride=1, padding=0
    const int conv1_out_w = LenetShape::in_width - LenetShape::conv1_kernel + 1;
    const int pool1_out_h = conv1_out_h / LenetShape::pool_stride;
    const int pool1_out_w = conv1_out_w / LenetShape::pool_stride;

    const int conv2_in_h = pool1_out_h;
    const int conv2_in_w = pool1_out_w;
    const int conv2_out_h = conv2_in_h - LenetShape::conv2_kernel + 1;
    const int conv2_out_w = conv2_in_w - LenetShape::conv2_kernel + 1;
    const int pool2_out_h = conv2_out_h / LenetShape::pool_stride;
    const int pool2_out_w = conv2_out_w / LenetShape::pool_stride;

    const int flattened = LenetShape::conv2_out_channels * pool2_out_h * pool2_out_w;

    s.input_elements = static_cast<size_t>(batch) * LenetShape::in_channels * LenetShape::in_height * LenetShape::in_width;
    s.conv1_out_elems = static_cast<size_t>(batch) * LenetShape::conv1_out_channels * conv1_out_h * conv1_out_w;
    s.pool1_out_elems = static_cast<size_t>(batch) * LenetShape::conv1_out_channels * pool1_out_h * pool1_out_w;
    s.conv2_out_elems = static_cast<size_t>(batch) * LenetShape::conv2_out_channels * conv2_out_h * conv2_out_w;
    s.pool2_out_elems = static_cast<size_t>(batch) * LenetShape::conv2_out_channels * pool2_out_h * pool2_out_w;
    s.fc1_out_elems = static_cast<size_t>(batch) * LenetShape::fc1_out;
    s.fc2_out_elems = static_cast<size_t>(batch) * LenetShape::fc2_out;
    s.output_elements = static_cast<size_t>(batch) * LenetShape::fc3_out;

    s.weight_offsets = std::vector<size_t>(5, 0);
    s.bias_offsets = std::vector<size_t>(5, 0);
    size_t cursor_w = 0;
    size_t cursor_b = 0;
    s.weight_offsets[0] = cursor_w;
    cursor_w += static_cast<size_t>(LenetShape::conv1_out_channels) * LenetShape::in_channels * LenetShape::conv1_kernel * LenetShape::conv1_kernel;
    s.weight_offsets[1] = cursor_w;
    cursor_w += static_cast<size_t>(LenetShape::conv2_out_channels) * LenetShape::conv1_out_channels * LenetShape::conv2_kernel * LenetShape::conv2_kernel;
    s.weight_offsets[2] = cursor_w;
    cursor_w += static_cast<size_t>(LenetShape::fc1_out) * flattened;
    s.weight_offsets[3] = cursor_w;
    cursor_w += static_cast<size_t>(LenetShape::fc2_out) * LenetShape::fc1_out;
    s.weight_offsets[4] = cursor_w;
    cursor_w += static_cast<size_t>(LenetShape::fc3_out) * LenetShape::fc2_out;

    s.bias_offsets[0] = cursor_b;
    cursor_b += LenetShape::conv1_out_channels;
    s.bias_offsets[1] = cursor_b;
    cursor_b += LenetShape::conv2_out_channels;
    s.bias_offsets[2] = cursor_b;
    cursor_b += LenetShape::fc1_out;
    s.bias_offsets[3] = cursor_b;
    cursor_b += LenetShape::fc2_out;
    s.bias_offsets[4] = cursor_b;
    cursor_b += LenetShape::fc3_out;

    s.total_weight_elements = cursor_w;
    s.total_bias_elements = cursor_b;
    return s;
}

inline double lenet_gflops(const LenetShape& shape, double millis) {
    const double conv1_flops = static_cast<double>(shape.batch) * LenetShape::conv1_out_channels * LenetShape::in_channels *
                               LenetShape::conv1_kernel * LenetShape::conv1_kernel * 2.0 * 28 * 28;
    const double conv2_flops = static_cast<double>(shape.batch) * LenetShape::conv2_out_channels * LenetShape::conv1_out_channels *
                               LenetShape::conv2_kernel * LenetShape::conv2_kernel * 2.0 * 10 * 10;
    const double fc1_in = LenetShape::conv2_out_channels * 5 * 5;
    const double fc_flops = static_cast<double>(shape.batch) *
                            (2.0 * fc1_in * LenetShape::fc1_out +
                             2.0 * LenetShape::fc1_out * LenetShape::fc2_out +
                             2.0 * LenetShape::fc2_out * LenetShape::fc3_out);
    const double total = conv1_flops + conv2_flops + fc_flops;
    return total / (millis * 1e6);
}

struct LenetDescriptors {
    cudnnTensorDescriptor_t input_desc = nullptr;
    cudnnTensorDescriptor_t conv1_out_desc = nullptr;
    cudnnTensorDescriptor_t pool1_out_desc = nullptr;
    cudnnTensorDescriptor_t conv2_out_desc = nullptr;
    cudnnTensorDescriptor_t pool2_out_desc = nullptr;
    cudnnTensorDescriptor_t fc1_desc = nullptr;
    cudnnTensorDescriptor_t fc2_desc = nullptr;
    cudnnTensorDescriptor_t fc3_desc = nullptr;

    cudnnFilterDescriptor_t conv1_filter = nullptr;
    cudnnFilterDescriptor_t conv2_filter = nullptr;

    cudnnConvolutionDescriptor_t conv1_desc = nullptr;
    cudnnConvolutionDescriptor_t conv2_desc = nullptr;

    cudnnActivationDescriptor_t activation = nullptr;
    cudnnPoolingDescriptor_t pool = nullptr;
};

inline void create_lenet_descriptors(const LenetShape& shape, LenetDescriptors& d) {
    const int B = shape.batch;
    const int conv1_h = LenetShape::in_height - LenetShape::conv1_kernel + 1;
    const int conv1_w = LenetShape::in_width  - LenetShape::conv1_kernel + 1;
    const int pool1_h = conv1_h / LenetShape::pool_stride;
    const int pool1_w = conv1_w / LenetShape::pool_stride;
    const int conv2_h = pool1_h - LenetShape::conv2_kernel + 1;
    const int conv2_w = pool1_w - LenetShape::conv2_kernel + 1;
    const int pool2_h = conv2_h / LenetShape::pool_stride;
    const int pool2_w = conv2_w / LenetShape::pool_stride;

    // Helper: create + configure a 4-D NCHW tensor descriptor.
    auto makeTensor = [](cudnnTensorDescriptor_t& desc, int n, int c, int h, int w) {
        cudnnCreateTensorDescriptor(&desc);
        cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    };
    makeTensor(d.input_desc,     B, LenetShape::in_channels,        LenetShape::in_height, LenetShape::in_width);
    makeTensor(d.conv1_out_desc, B, LenetShape::conv1_out_channels, conv1_h, conv1_w);
    makeTensor(d.pool1_out_desc, B, LenetShape::conv1_out_channels, pool1_h, pool1_w);
    makeTensor(d.conv2_out_desc, B, LenetShape::conv2_out_channels, conv2_h, conv2_w);
    makeTensor(d.pool2_out_desc, B, LenetShape::conv2_out_channels, pool2_h, pool2_w);
    makeTensor(d.fc1_desc, B, LenetShape::fc1_out, 1, 1);
    makeTensor(d.fc2_desc, B, LenetShape::fc2_out, 1, 1);
    makeTensor(d.fc3_desc, B, LenetShape::fc3_out, 1, 1);

    // Filter descriptors.
    auto makeFilter = [](cudnnFilterDescriptor_t& desc, int k, int c, int h, int w) {
        cudnnCreateFilterDescriptor(&desc);
        cudnnSetFilter4dDescriptor(desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, h, w);
    };
    makeFilter(d.conv1_filter, LenetShape::conv1_out_channels, LenetShape::in_channels,
               LenetShape::conv1_kernel, LenetShape::conv1_kernel);
    makeFilter(d.conv2_filter, LenetShape::conv2_out_channels, LenetShape::conv1_out_channels,
               LenetShape::conv2_kernel, LenetShape::conv2_kernel);

    // Convolution descriptors (pad=0, stride=1, dilation=1, cross-correlation).
    auto makeConv = [](cudnnConvolutionDescriptor_t& desc) {
        cudnnCreateConvolutionDescriptor(&desc);
        cudnnSetConvolution2dDescriptor(desc, 0, 0, 1, 1, 1, 1,
                                        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    };
    makeConv(d.conv1_desc);
    makeConv(d.conv2_desc);

    // Activation descriptor (tanh — classic LeNet-5).
    cudnnCreateActivationDescriptor(&d.activation);
    cudnnSetActivationDescriptor(d.activation, CUDNN_ACTIVATION_TANH,
                                 CUDNN_PROPAGATE_NAN, 0.0);

    // Pooling descriptor (max pool 2×2, stride 2).
    cudnnCreatePoolingDescriptor(&d.pool);
    cudnnSetPooling2dDescriptor(d.pool, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_PROPAGATE_NAN,
                                LenetShape::pool_stride, LenetShape::pool_stride,
                                0, 0,
                                LenetShape::pool_stride, LenetShape::pool_stride);
}

inline void destroy_lenet_descriptors(LenetDescriptors& d) {
    if (d.pool)          cudnnDestroyPoolingDescriptor(d.pool);
    if (d.activation)    cudnnDestroyActivationDescriptor(d.activation);
    if (d.conv2_desc)    cudnnDestroyConvolutionDescriptor(d.conv2_desc);
    if (d.conv1_desc)    cudnnDestroyConvolutionDescriptor(d.conv1_desc);
    if (d.conv2_filter)  cudnnDestroyFilterDescriptor(d.conv2_filter);
    if (d.conv1_filter)  cudnnDestroyFilterDescriptor(d.conv1_filter);
    if (d.fc3_desc)      cudnnDestroyTensorDescriptor(d.fc3_desc);
    if (d.fc2_desc)      cudnnDestroyTensorDescriptor(d.fc2_desc);
    if (d.fc1_desc)      cudnnDestroyTensorDescriptor(d.fc1_desc);
    if (d.pool2_out_desc) cudnnDestroyTensorDescriptor(d.pool2_out_desc);
    if (d.conv2_out_desc) cudnnDestroyTensorDescriptor(d.conv2_out_desc);
    if (d.pool1_out_desc) cudnnDestroyTensorDescriptor(d.pool1_out_desc);
    if (d.conv1_out_desc) cudnnDestroyTensorDescriptor(d.conv1_out_desc);
    if (d.input_desc)    cudnnDestroyTensorDescriptor(d.input_desc);
}

inline cudnnConvolutionFwdAlgo_t parse_algo(const std::string& name) {
    if (name == "implicit_gemm") return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    if (name == "implicit_precomp") return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    if (name == "fft") return CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    if (name == "fft_tiling") return CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
    if (name == "direct") return CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
    if (name == "gemm") return CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
    if (name == "winograd") return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    if (name == "winograd_nonfused") return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
}

inline size_t query_conv_workspace(cudnnHandle_t handle,
                                   const LenetShape& shape,
                                   const LenetDescriptors& descs,
                                   cudnnConvolutionFwdAlgo_t algo,
                                   bool second_conv) {
    (void)shape;
    size_t bytes = 0;
    if (!second_conv) {
        cudnnGetConvolutionForwardWorkspaceSize(handle,
            descs.input_desc, descs.conv1_filter,
            descs.conv1_desc, descs.conv1_out_desc, algo, &bytes);
    } else {
        cudnnGetConvolutionForwardWorkspaceSize(handle,
            descs.pool1_out_desc, descs.conv2_filter,
            descs.conv2_desc, descs.conv2_out_desc, algo, &bytes);
    }
    return bytes;
}

inline void run_lenet_conv(cudnnHandle_t handle,
                           const LenetShape& shape,
                           const LenetDescriptors& descs,
                           const float* d_input,
                           const float* d_filter,
                           float* d_output,
                           void* d_workspace,
                           size_t workspace_bytes,
                           const std::string& algo_name,
                           bool second_conv) {
    (void)shape;
    const float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionFwdAlgo_t algo = parse_algo(algo_name);

    if (!second_conv) {
        cudnnConvolutionForward(handle, &alpha,
            descs.input_desc, d_input,
            descs.conv1_filter, d_filter,
            descs.conv1_desc, algo,
            d_workspace, workspace_bytes,
            &beta, descs.conv1_out_desc, d_output);
    } else {
        cudnnConvolutionForward(handle, &alpha,
            descs.pool1_out_desc, d_input,
            descs.conv2_filter, d_filter,
            descs.conv2_desc, algo,
            d_workspace, workspace_bytes,
            &beta, descs.conv2_out_desc, d_output);
    }
}

inline void run_lenet_pool(cudnnHandle_t handle,
                           const LenetDescriptors& descs,
                           const float* d_input,
                           float* d_output,
                           bool second_pool) {
    const float alpha = 1.0f, beta = 0.0f;
    if (!second_pool) {
        cudnnPoolingForward(handle, descs.pool, &alpha,
            descs.conv1_out_desc, d_input,
            &beta, descs.pool1_out_desc, d_output);
    } else {
        cudnnPoolingForward(handle, descs.pool, &alpha,
            descs.conv2_out_desc, d_input,
            &beta, descs.pool2_out_desc, d_output);
    }
}

__global__ void bias_tanh_kernel(float* data, const float* bias, int total, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx] = tanhf(data[idx] + bias[idx % cols]);
    }
}

__global__ void bias_only_kernel(float* data, const float* bias, int total, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx] += bias[idx % cols];
    }
}

inline void run_fc_layer(cublasHandle_t handle,
                         const LenetShape& shape,
                         int layer_idx,
                         const float* d_input,
                         const float* d_weight,
                         const float* d_bias,
                         float* d_output,
                         cudaStream_t stream) {
    const int B = shape.batch;
    int K, N;
    if (layer_idx == 0) {
        K = LenetShape::conv2_out_channels * 5 * 5;  // 400
        N = LenetShape::fc1_out;                      // 120
    } else if (layer_idx == 1) {
        K = LenetShape::fc1_out;                      // 120
        N = LenetShape::fc2_out;                      // 84
    } else {
        K = LenetShape::fc2_out;                      // 84
        N = LenetShape::fc3_out;                      // 10
    }

    const float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(handle, stream);
    // Row-major C(B,N) = A(B,K) * W^T  ↔  col-major C'(N,B) = W * A'
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, B, K,
                &alpha, d_weight, K,
                d_input, K,
                &beta, d_output, N);

    const int total = B * N;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    if (layer_idx < 2) {
        // fc1, fc2: bias + tanh activation
        bias_tanh_kernel<<<blocks, threads, 0, stream>>>(d_output, d_bias, total, N);
    } else {
        // fc3 (output layer): bias only, no activation
        bias_only_kernel<<<blocks, threads, 0, stream>>>(d_output, d_bias, total, N);
    }
}

inline void reshape_conv_to_fc(const LenetShape& shape, const float* d_input, float* d_output, cudaStream_t stream) {
    // NCHW is already contiguous per batch element, so (B,C,H,W) == (B, C*H*W) in memory.
    cudaMemcpyAsync(d_output, d_input,
                    shape.pool2_out_elems * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
}
