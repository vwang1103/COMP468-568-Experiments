#pragma once

#include <cuda_runtime.h>

struct Conv2dShape {
    int height;
    int width;
    int channels;
    int filters;
    int kernel;
    int stride;
    int padding;
    int out_height;
    int out_width;
};

inline Conv2dShape make_shape(int height,
                              int width,
                              int channels,
                              int filters,
                              int kernel,
                              int stride,
                              int padding) {
    Conv2dShape shape{height,
                      width,
                      channels,
                      filters,
                      kernel,
                      stride,
                      padding,
                      0,
                      0};
    shape.out_height = (height + 2 * padding - kernel) / stride + 1;
    shape.out_width = (width + 2 * padding - kernel) / stride + 1;
    return shape;
}

inline __host__ __device__ int input_index(const Conv2dShape& shape, int c, int h, int w) {
    return (c * shape.height + h) * shape.width + w;
}

inline __host__ __device__ int weight_index(const Conv2dShape& shape, int oc, int ic, int kh, int kw) {
    return ((oc * shape.channels + ic) * shape.kernel + kh) * shape.kernel + kw;
}

inline __host__ __device__ int output_index(const Conv2dShape& shape, int oc, int oh, int ow) {
    return (oc * shape.out_height + oh) * shape.out_width + ow;
}

constexpr int BLOCK_SIZE = 16;
constexpr int TILE_C = 8;  // input channels batched per shared-memory load in tiled kernel

inline dim3 make_conv_grid(const Conv2dShape& shape) {
    return dim3((shape.out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (shape.out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
                shape.filters);
}

__global__ void conv2d_naive_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    float* __restrict__ output,
                                    Conv2dShape shape) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int oc = blockIdx.z;
    if (ow >= shape.out_width || oh >= shape.out_height || oc >= shape.filters) {
        return;
    }

    /* TODO(student): loop over channels/ksize and accumulate into acc. Remember padding offsets:
       ih = oh * stride - padding + kh;
       iw = ow * stride - padding + kw;
       Skip taps that fall outside the padded image. */
    const int H = shape.height;
    const int W = shape.width;
    const int C = shape.channels;
    const int K = shape.kernel;
    const int ih_base = oh * shape.stride - shape.padding;
    const int iw_base = ow * shape.stride - shape.padding;

    // Pre-clamp loop bounds to eliminate all inner-loop branches
    const int kh_start = max(0, -ih_base);
    const int kh_end   = min(K, H - ih_base);
    const int kw_start = max(0, -iw_base);
    const int kw_end   = min(K, W - iw_base);

    float acc = 0.0f;
    for (int ic = 0; ic < C; ++ic) {
        const int in_c_off = ic * H * W;
        const int w_ic_off = (oc * C + ic) * K * K;
        #pragma unroll
        for (int kh = kh_start; kh < kh_end; ++kh) {
            const int in_row_off = in_c_off + (ih_base + kh) * W;
            const int w_row_off = w_ic_off + kh * K;
            #pragma unroll
            for (int kw = kw_start; kw < kw_end; ++kw) {
                acc += __ldg(&input[in_row_off + iw_base + kw]) *
                       __ldg(&weight[w_row_off + kw]);
            }
        }
    }
    output[output_index(shape, oc, oh, ow)] = acc;
}

__global__ void conv2d_tiled_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    float* __restrict__ output,
                                    Conv2dShape shape) {
    /* TODO(student): stage input patches / filter tiles into shared memory and reuse them
       across threads within the block. Carefully handle halos caused by padding. */
    const int ow = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int oh = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int oc = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int K = shape.kernel;
    const int H = shape.height;
    const int W = shape.width;
    const int C = shape.channels;
    const int S = shape.stride;
    const int P = shape.padding;

    const int tile_h = BLOCK_SIZE + K - 1;
    const int tile_w = BLOCK_SIZE + K - 1;
    const int tile_spatial = tile_h * tile_w;
    const int wt_size = K * K;

    extern __shared__ float smem[];
    // Layout: [TILE_C * tile_spatial] for input, then [TILE_C * K*K] for weights
    float* s_input  = smem;
    float* s_weight = smem + TILE_C * tile_spatial;

    float acc = 0.0f;

    const int base_ih = blockIdx.y * BLOCK_SIZE * S - P;
    const int base_iw = blockIdx.x * BLOCK_SIZE * S - P;
    const int tid = ty * BLOCK_SIZE + tx;
    const int threads = BLOCK_SIZE * BLOCK_SIZE;
    const bool in_bounds = (ow < shape.out_width && oh < shape.out_height);
    const int sh = ty * S;
    const int sw = tx * S;

    // Process TILE_C input channels per shared-memory load to amortize __syncthreads cost
    for (int ic_start = 0; ic_start < C; ic_start += TILE_C) {
        const int nc = min(TILE_C, C - ic_start);

        // Cooperatively load nc channels' input patches into shared memory
        const int total_input = nc * tile_spatial;
        for (int idx = tid; idx < total_input; idx += threads) {
            const int c_local = idx / tile_spatial;
            const int s_idx = idx - c_local * tile_spatial;
            const int local_h = s_idx / tile_w;
            const int local_w = s_idx - local_h * tile_w;
            const int ih = base_ih + local_h;
            const int iw = base_iw + local_w;
            const int ic = ic_start + c_local;
            float val = 0.0f;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W && ic < C) {
                val = __ldg(&input[(ic * H + ih) * W + iw]);
            }
            s_input[idx] = val;
        }

        // Cooperatively load nc channels' weight tiles for this output filter
        const int total_wt = nc * wt_size;
        for (int idx = tid; idx < total_wt; idx += threads) {
            const int c_local = idx / wt_size;
            const int w_idx = idx - c_local * wt_size;
            const int ic = ic_start + c_local;
            s_weight[idx] = (ic < C) ? __ldg(&weight[((oc * C + ic) * wt_size) + w_idx]) : 0.0f;
        }
        __syncthreads();

        // Accumulate from shared memory (both input and weights)
        if (in_bounds) {
            for (int c_local = 0; c_local < nc; ++c_local) {
                const float* tile = s_input + c_local * tile_spatial;
                const float* wt   = s_weight + c_local * wt_size;
                #pragma unroll
                for (int kh = 0; kh < K; ++kh) {
                    const int tile_row = (sh + kh) * tile_w + sw;
                    const int w_row = kh * K;
                    #pragma unroll
                    for (int kw = 0; kw < K; ++kw) {
                        acc += tile[tile_row + kw] * wt[w_row + kw];
                    }
                }
            }
        }
        __syncthreads();
    }

    if (in_bounds && oc < shape.filters) {
        output[(oc * shape.out_height + oh) * shape.out_width + ow] = acc;
    }
}

inline void launch_naive_conv2d(const float* d_input,
                                const float* d_weight,
                                float* d_output,
                                const Conv2dShape& shape,
                                cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid = make_conv_grid(shape);
    conv2d_naive_kernel<<<grid, block, 0, stream>>>(d_input, d_weight, d_output, shape);
    /* TODO(student): check cudaGetLastError() and optionally cudaDeviceSynchronize() when debugging. */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("naive kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

inline void launch_tiled_conv2d(const float* d_input,
                                const float* d_weight,
                                float* d_output,
                                const Conv2dShape& shape,
                                cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid = make_conv_grid(shape);
    /* TODO(student): choose a better shared-memory layout/size expression once kernels are implemented. */
    const int tile_h = BLOCK_SIZE + shape.kernel - 1;
    const int tile_w = BLOCK_SIZE + shape.kernel - 1;
    int nc = min(TILE_C, shape.channels);
    // Input tiles + weight tiles in shared memory
    size_t shared_bytes = static_cast<size_t>(nc) * (tile_h * tile_w + shape.kernel * shape.kernel) * sizeof(float);
    conv2d_tiled_kernel<<<grid, block, shared_bytes, stream>>>(d_input, d_weight, d_output, shape);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("tiled kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

inline double conv_gflops(const Conv2dShape& shape, double millis) {
    const double flops = static_cast<double>(shape.filters) * shape.out_height * shape.out_width *
                         shape.channels * shape.kernel * shape.kernel * 2.0;
    return flops / (millis * 1e6);
}