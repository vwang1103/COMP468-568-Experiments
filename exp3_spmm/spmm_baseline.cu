// spmm_baseline.cu — STUDENT SKELETON
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>

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

using float_t = float;

#define CUDA_CHECK(call) do {                                        \
    cudaError_t err__ = (call);                                       \
    if (err__ != cudaSuccess) {                                       \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__)      \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";   \
        std::exit(1);                                                 \
    }                                                                 \
} while (0)

/*
===============================================================
 BASELINE KERNEL — one thread processes ONE ROW of A
 STUDENT TODO: 
   - Fill missing loops
   - Compute C[row, j] += value * B[k, j]
===============================================================
*/
__global__ void spmm_csr_row_kernel(
    int M, int N,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float_t* __restrict__ d_vals,
    const float_t* __restrict__ d_B,
    float_t* __restrict__ d_C) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // TODO (student): Initialize output row C[row, :]
    float_t* Crow = d_C + (size_t)row * N;
    for (int j = 0; j < N; ++j) {
        Crow[j] = 0.0f;
    }

    // Find nonzero range
    int start, end;
    // TODO (student): load start, end 
    start = d_row_ptr[row];
    end   = d_row_ptr[row + 1];

    // Loop over nonzeros in this row
    // TODO (student): 
    for (int p = start; p < end; ++p)
    {
        // TODO (student): retrieve column index k 
        int k = d_col_idx[p];
        // TODO (student): retrieve value v 
        float_t v = d_vals[p];

        // TODO (student): loop over all columns j of output (0..N-1)
        //                 and accumulate:
        const float_t* Brow = d_B + (size_t)k * N;
        for (int j = 0; j < N; ++j) {
            Crow[j] += v * Brow[j];
        }
    }
}

/*
===============================================================
 MAIN PROGRAM
===============================================================
*/
int main(int argc, char** argv) {
    // Defaults (can override via CLI):
    //   ./spmm_baseline [M K N density iters]
    int M = 4096, K = 4096, N = 256;
    double density = 0.01;
    unsigned seed = 1234;
    int iters = 200;

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
    int *d_row_ptr, *d_col_idx;
    float *d_vals, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (M+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)K*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M*N*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (M+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, vals.data(), nnz*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), (size_t)K*N*sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (M + block - 1) / block;

    // ------------------------------------------------------------
    // Timing + GFLOP/s
    // ------------------------------------------------------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    spmm_csr_row_kernel<<<grid, block>>>(M, N, d_row_ptr, d_col_idx, d_vals, d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // timed runs
    CUDA_CHECK(cudaEventRecord(start));
    for (int t = 0; t < iters; ++t) {
        spmm_csr_row_kernel<<<grid, block>>>(M, N, d_row_ptr, d_col_idx, d_vals, d_B, d_C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iters;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // FLOPs for SpMM: 2 * nnz * N  (mul + add per output column)
    double flops = 2.0 * (double)nnz * (double)N;
    double gflops = flops / ((avg_ms / 1e3) * 1e9);

    std::cout << "Avg kernel time (ms) = " << avg_ms << "\n";
    std::cout << "GFLOP/s = " << gflops << "\n";

    // Copy back (from last timed run)
    std::vector<float> C((size_t)M*N);
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, (size_t)M*N*sizeof(float), cudaMemcpyDeviceToHost));

    // Compare against CPU reference
    float err = max_abs_err(C_ref, C);
    std::cout << "Max error = " << err << "\n";

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}