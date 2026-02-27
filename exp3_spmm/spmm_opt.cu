// spmm_opt.cu — STUDENT OPTIMIZATION SKELETON

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>

using float_t = float;

// Reference functions provided by spmm_ref.cpp
extern void generate_random_csr(int M, int K, double density,
                                std::vector<int>& row_ptr,
                                std::vector<int>& col_idx,
                                std::vector<float>& vals,
                                unsigned seed);

extern void spmm_cpu(int M, int K, int N,
                     const std::vector<int>& row_ptr,
                     const std::vector<int>& col_idx,
                     const std::vector<float>& vals,
                     const std::vector<float>& B,
                     std::vector<float>& C);

extern float max_abs_err(const std::vector<float>& A, const std::vector<float>& B);

/*
=================================================================
 OPTIMIZED KERNEL (SKELETON)
 Warp processes ONE ROW, each thread handles j = lane, lane+32, ...
 STUDENT TODO:
    - Fetch row range
    - Loop over nonzeros
    - Load B[k,j] and accumulate
=================================================================
*/
__global__ void spmm_csr_warp_kernel(
    int M, int N,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float_t* __restrict__ d_vals,
    const float_t* __restrict__ d_B,
    float_t* __restrict__ d_C)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = global_tid / 32;
    int lane = threadIdx.x % 32;

    if (warp >= M) return;
    int row = warp;

    // TODO (student): get start = d_row_ptr[row], end = d_row_ptr[row+1]
    int start = d_row_ptr[row];
    int end   = d_row_ptr[row + 1];

    // Loop over columns j assigned to this lane
    for (int j = lane; j < N; j += 32) {

        float_t sum = 0.0f;

        // TODO (student): loop over nonzeros in this row
        for (int p = start; p < end; ++p) {
            int k = d_col_idx[p];
            float_t v = d_vals[p];
            sum += v * d_B[(size_t)k * N + j];
        }

        // TODO (student): write result to d_C
        d_C[(size_t)row * N + j] = sum;
    }
}

/*
===========================================================
 MAIN DRIVER
===========================================================
*/
int main(int argc, char** argv) {
    // Defaults (can override via CLI):
    //   ./spmm_opt [M K N density iters]
    int M = 512, K = 512, N = 64;
    double density = 0.01;
    int iters = 200;
    unsigned seed = 1234;

    if (argc >= 5) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
        density = std::atof(argv[4]);
    }

    std::vector<int> row_ptr, col_idx;
    std::vector<float> vals;
    generate_random_csr(M, K, density, row_ptr, col_idx, vals, seed);
    int nnz = row_ptr.back();

    std::cout << "nnz = " << nnz << "\n";

    // Create B
    std::vector<float> B((size_t)K * N);
    for (size_t i = 0; i < B.size(); i++) B[i] = float(rand()) / RAND_MAX;

    // CPU reference
    std::vector<float> C_ref;
    spmm_cpu(M, K, N, row_ptr, col_idx, vals, B, C_ref);

    // Copy to device
    int *d_row_ptr = nullptr, *d_col_idx = nullptr;
    float *d_vals = nullptr, *d_B = nullptr, *d_C = nullptr;

    cudaMalloc(&d_row_ptr, (M+1)*sizeof(int));
    cudaMalloc(&d_col_idx, nnz*sizeof(int));
    cudaMalloc(&d_vals, nnz*sizeof(float));
    cudaMalloc(&d_B, (size_t)K*N*sizeof(float));
    cudaMalloc(&d_C, (size_t)M*N*sizeof(float));

    cudaMemcpy(d_row_ptr, row_ptr.data(), (M+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, vals.data(), nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), (size_t)K*N*sizeof(float), cudaMemcpyHostToDevice);

    // Launch config: need M warps total (1 warp per row)
    int block = 256; // 8 warps per block
    long long total_threads_needed = 1LL * M * 32;
    int grid = (int)((total_threads_needed + block - 1) / block);

    // ------------------------------------------------------------
    // Warmup + timed kernel launches (kernel time only)
    // ------------------------------------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    spmm_csr_warp_kernel<<<grid, block>>>(M, N, d_row_ptr, d_col_idx, d_vals, d_B, d_C);
    cudaDeviceSynchronize();

    // Timed runs
    cudaEventRecord(start);
    for (int t = 0; t < iters; ++t) {
        spmm_csr_warp_kernel<<<grid, block>>>(M, N, d_row_ptr, d_col_idx, d_vals, d_B, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / iters;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // GFLOP/s for SpMM: 2 * nnz * N FLOPs
    double flops = 2.0 * (double)nnz * (double)N;
    double gflops = flops / ((avg_ms / 1e3) * 1e9);

    std::cout << "Avg kernel time (ms) = " << avg_ms << "\n";
    std::cout << "GFLOP/s = " << gflops << "\n";

    // Copy result back
    std::vector<float> C((size_t)M * N);
    cudaMemcpy(C.data(), d_C, (size_t)M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Correctness check vs CPU reference
    float err = max_abs_err(C_ref, C);
    std::cout << "Max error = " << err << "\n";

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_vals);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}