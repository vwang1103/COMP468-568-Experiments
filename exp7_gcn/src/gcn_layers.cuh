#pragma once

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct GraphData {
    int num_nodes = 0;
    int num_edges = 0;  // number of directed entries from CSR
    int nnz = 0;        // CSR nnz (match file; do NOT add self loops here)
    int feature_dim = 0;
    int num_classes = 0;

    std::vector<int> h_csr_row_offsets;
    std::vector<int> h_csr_col_indices;
    std::vector<float> h_csr_values;
    std::vector<float> h_features;   // row-major: num_nodes x feature_dim
    std::vector<int> h_labels;       // node labels
};

struct DeviceGCNWorkspace {
    int* d_csr_row_offsets = nullptr;
    int* d_csr_col_indices = nullptr;
    float* d_csr_values = nullptr;

    float* d_features_in = nullptr;
    float* d_features_out = nullptr;
    float* d_weights = nullptr;
    float* d_logits = nullptr;
    float* d_temp = nullptr;

    cusparseSpMatDescr_t spmat = nullptr;
    cusparseDnMatDescr_t dn_left = nullptr;
    cusparseDnMatDescr_t dn_right = nullptr;
    cusparseDnMatDescr_t dn_out = nullptr;
};

inline void check_cusparse(cusparseStatus_t status, const char* msg) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuSPARSE error");
    }
}

inline void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuBLAS error");
    }
}

inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(err));
    }
}

inline std::size_t get_file_size_bytes(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    return static_cast<std::size_t>(ifs.tellg());
}

template <typename T>
inline void read_binary_exact(const std::string& path, std::vector<T>& out, std::size_t count) {
    out.resize(count);
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    ifs.read(reinterpret_cast<char*>(out.data()),
             static_cast<std::streamsize>(count * sizeof(T)));
    if (!ifs) {
        throw std::runtime_error("Failed to read expected bytes from file: " + path);
    }
}

inline void build_graph_from_files(const std::string& prefix, GraphData& graph) {
    const std::string csr_path = prefix + ".csr";
    const std::string feat_path = prefix + ".feat";
    const std::string label_path = prefix + ".label";

    // Read CSR header
    std::ifstream csr_ifs(csr_path, std::ios::binary);
    if (!csr_ifs) {
        throw std::runtime_error("Failed to open CSR file: " + csr_path);
    }

    int num_nodes = 0;
    int nnz_original = 0;
    csr_ifs.read(reinterpret_cast<char*>(&num_nodes), sizeof(int));
    csr_ifs.read(reinterpret_cast<char*>(&nnz_original), sizeof(int));
    if (!csr_ifs) {
        throw std::runtime_error("Failed to read CSR header from: " + csr_path);
    }
    if (num_nodes <= 0 || nnz_original < 0) {
        throw std::runtime_error("Invalid CSR header in: " + csr_path);
    }

    std::vector<int> row_offsets(num_nodes + 1);
    std::vector<int> col_indices(nnz_original);

    csr_ifs.read(reinterpret_cast<char*>(row_offsets.data()),
                 static_cast<std::streamsize>((num_nodes + 1) * sizeof(int)));
    csr_ifs.read(reinterpret_cast<char*>(col_indices.data()),
                 static_cast<std::streamsize>(nnz_original * sizeof(int)));
    if (!csr_ifs) {
        throw std::runtime_error("Failed to read CSR payload from: " + csr_path);
    }
    csr_ifs.close();

    // Infer feature_dim from file size
    const std::size_t feat_bytes = get_file_size_bytes(feat_path);
    if (feat_bytes % sizeof(float) != 0) {
        throw std::runtime_error("Feature file size is not divisible by sizeof(float): " + feat_path);
    }
    const std::size_t feat_count = feat_bytes / sizeof(float);
    if (feat_count % static_cast<std::size_t>(num_nodes) != 0) {
        throw std::runtime_error("Feature count is not divisible by num_nodes in: " + feat_path);
    }
    const int feature_dim = static_cast<int>(feat_count / static_cast<std::size_t>(num_nodes));
    if (feature_dim <= 0) {
        throw std::runtime_error("Inferred invalid feature_dim from: " + feat_path);
    }

    // Read features
    read_binary_exact<float>(feat_path, graph.h_features, feat_count);

    // Read labels
    const std::size_t label_bytes = get_file_size_bytes(label_path);
    if (label_bytes % sizeof(int) != 0) {
        throw std::runtime_error("Label file size is not divisible by sizeof(int): " + label_path);
    }
    const std::size_t label_count = label_bytes / sizeof(int);
    if (label_count != static_cast<std::size_t>(num_nodes)) {
        throw std::runtime_error("Label count does not match num_nodes in: " + label_path);
    }
    read_binary_exact<int>(label_path, graph.h_labels, label_count);

    int max_label = 0;
    for (int y : graph.h_labels) {
        if (y < 0) {
            throw std::runtime_error("Negative label found in: " + label_path);
        }
        max_label = std::max(max_label, y);
    }

    // Match DGL GraphConv(norm='both') on the raw directed graph from file:
    //   weight(src -> dst) = 1 / sqrt(outdeg(src)) / sqrt(indeg(dst))
    // Do NOT add self-loops here.
    std::vector<float> out_deg(num_nodes, 0.0f);
    std::vector<float> in_deg(num_nodes, 0.0f);

    for (int src = 0; src < num_nodes; ++src) {
        const int start = row_offsets[src];
        const int end = row_offsets[src + 1];
        if (start > end || start < 0 || end > nnz_original) {
            throw std::runtime_error("Invalid CSR row range");
        }

        out_deg[src] = static_cast<float>(end - start);

        for (int idx = start; idx < end; ++idx) {
            const int dst = col_indices[idx];
            if (dst < 0 || dst >= num_nodes) {
                throw std::runtime_error("CSR column index out of range");
            }
            in_deg[dst] += 1.0f;
        }
    }

    graph.h_csr_row_offsets = std::move(row_offsets);
    graph.h_csr_col_indices = std::move(col_indices);
    graph.h_csr_values.resize(static_cast<std::size_t>(nnz_original));

    for (int src = 0; src < num_nodes; ++src) {
        const int start = graph.h_csr_row_offsets[src];
        const int end = graph.h_csr_row_offsets[src + 1];

        for (int idx = start; idx < end; ++idx) {
            const int dst = graph.h_csr_col_indices[idx];

            const float od = out_deg[src];
            const float id = in_deg[dst];

            float val = 0.0f;
            if (od > 0.0f && id > 0.0f) {
                val = 1.0f / std::sqrt(od * id);
            }
            graph.h_csr_values[idx] = val;
        }
    }

    graph.num_nodes = num_nodes;
    graph.num_edges = nnz_original;
    graph.nnz = nnz_original;
    graph.feature_dim = feature_dim;
    graph.num_classes = max_label + 1;
}

inline void allocate_device_graph(const GraphData& graph, DeviceGCNWorkspace& workspace) {
    check_cuda(cudaMalloc(&workspace.d_csr_row_offsets,
                          static_cast<std::size_t>(graph.num_nodes + 1) * sizeof(int)),
               "cudaMalloc d_csr_row_offsets");
    check_cuda(cudaMalloc(&workspace.d_csr_col_indices,
                          static_cast<std::size_t>(graph.nnz) * sizeof(int)),
               "cudaMalloc d_csr_col_indices");
    check_cuda(cudaMalloc(&workspace.d_csr_values,
                          static_cast<std::size_t>(graph.nnz) * sizeof(float)),
               "cudaMalloc d_csr_values");

    check_cuda(cudaMemcpy(workspace.d_csr_row_offsets,
                          graph.h_csr_row_offsets.data(),
                          static_cast<std::size_t>(graph.num_nodes + 1) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "copy row offsets");
    check_cuda(cudaMemcpy(workspace.d_csr_col_indices,
                          graph.h_csr_col_indices.data(),
                          static_cast<std::size_t>(graph.nnz) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "copy col indices");
    check_cuda(cudaMemcpy(workspace.d_csr_values,
                          graph.h_csr_values.data(),
                          static_cast<std::size_t>(graph.nnz) * sizeof(float),
                          cudaMemcpyHostToDevice),
               "copy csr values");

    const std::size_t feature_elems =
        static_cast<std::size_t>(graph.num_nodes) * graph.feature_dim;
    const std::size_t logits_elems =
        static_cast<std::size_t>(graph.num_nodes) * graph.num_classes;

    check_cuda(cudaMalloc(&workspace.d_features_in, feature_elems * sizeof(float)),
               "cudaMalloc d_features_in");
    check_cuda(cudaMalloc(&workspace.d_features_out, feature_elems * sizeof(float)),
               "cudaMalloc d_features_out");
    check_cuda(cudaMalloc(&workspace.d_logits, logits_elems * sizeof(float)),
               "cudaMalloc d_logits");
    check_cuda(cudaMalloc(&workspace.d_temp, feature_elems * sizeof(float)),
               "cudaMalloc d_temp");

    check_cuda(cudaMemcpy(workspace.d_features_in,
                          graph.h_features.data(),
                          feature_elems * sizeof(float),
                          cudaMemcpyHostToDevice),
               "copy input features");

    check_cusparse(
        cusparseCreateCsr(&workspace.spmat,
                          graph.num_nodes,
                          graph.num_nodes,
                          graph.nnz,
                          workspace.d_csr_row_offsets,
                          workspace.d_csr_col_indices,
                          workspace.d_csr_values,
                          CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO,
                          CUDA_R_32F),
        "cusparseCreateCsr");
}

inline void destroy_device_graph(DeviceGCNWorkspace& workspace) {
    if (workspace.dn_left) {
        cusparseDestroyDnMat(workspace.dn_left);
        workspace.dn_left = nullptr;
    }
    if (workspace.dn_right) {
        cusparseDestroyDnMat(workspace.dn_right);
        workspace.dn_right = nullptr;
    }
    if (workspace.dn_out) {
        cusparseDestroyDnMat(workspace.dn_out);
        workspace.dn_out = nullptr;
    }
    if (workspace.spmat) {
        cusparseDestroySpMat(workspace.spmat);
        workspace.spmat = nullptr;
    }

    cudaFree(workspace.d_csr_row_offsets);
    cudaFree(workspace.d_csr_col_indices);
    cudaFree(workspace.d_csr_values);
    cudaFree(workspace.d_features_in);
    cudaFree(workspace.d_features_out);
    cudaFree(workspace.d_weights);
    cudaFree(workspace.d_logits);
    cudaFree(workspace.d_temp);

    workspace.d_csr_row_offsets = nullptr;
    workspace.d_csr_col_indices = nullptr;
    workspace.d_csr_values = nullptr;
    workspace.d_features_in = nullptr;
    workspace.d_features_out = nullptr;
    workspace.d_weights = nullptr;
    workspace.d_logits = nullptr;
    workspace.d_temp = nullptr;
}

inline void run_sparse_dense_mm(cusparseHandle_t handle,
                                DeviceGCNWorkspace& workspace,
                                int rows,
                                int cols,
                                int K,
                                const float* d_input,
                                float* d_output) {
    (void)K;

    if (workspace.dn_right) {
        check_cusparse(cusparseDestroyDnMat(workspace.dn_right), "destroy old dn_right");
        workspace.dn_right = nullptr;
    }
    if (workspace.dn_out) {
        check_cusparse(cusparseDestroyDnMat(workspace.dn_out), "destroy old dn_out");
        workspace.dn_out = nullptr;
    }

    check_cusparse(
        cusparseCreateDnMat(&workspace.dn_right,
                            rows,
                            cols,
                            cols,
                            const_cast<float*>(d_input),
                            CUDA_R_32F,
                            CUSPARSE_ORDER_ROW),
        "cusparseCreateDnMat input");

    check_cusparse(
        cusparseCreateDnMat(&workspace.dn_out,
                            rows,
                            cols,
                            cols,
                            d_output,
                            CUDA_R_32F,
                            CUSPARSE_ORDER_ROW),
        "cusparseCreateDnMat output");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::size_t buffer_size = 0;
    check_cusparse(
        cusparseSpMM_bufferSize(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                workspace.spmat,
                                workspace.dn_right,
                                &beta,
                                workspace.dn_out,
                                CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT,
                                &buffer_size),
        "cusparseSpMM_bufferSize");

    void* d_buffer = nullptr;
    check_cuda(cudaMalloc(&d_buffer, buffer_size), "cudaMalloc SpMM buffer");

    check_cusparse(
        cusparseSpMM(handle,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha,
                     workspace.spmat,
                     workspace.dn_right,
                     &beta,
                     workspace.dn_out,
                     CUDA_R_32F,
                     CUSPARSE_SPMM_ALG_DEFAULT,
                     d_buffer),
        "cusparseSpMM");

    check_cuda(cudaFree(d_buffer), "cudaFree SpMM buffer");
}

inline void run_dense_layer(cublasHandle_t handle,
                            int M,
                            int K,
                            int N,
                            const float* d_input,
                            const float* d_weight,
                            float* d_output) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    check_cublas(
        cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    N,
                    M,
                    K,
                    &alpha,
                    d_weight,
                    N,
                    d_input,
                    K,
                    &beta,
                    d_output,
                    N),
        "cublasSgemm");
}

__global__ void relu_kernel(float* d_tensor, int elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) {
        d_tensor[idx] = fmaxf(0.0f, d_tensor[idx]);
    }
}

inline void apply_activation(float* d_tensor, int elements, cudaStream_t stream) {
    const int block = 256;
    const int grid = (elements + block - 1) / block;
    relu_kernel<<<grid, block, 0, stream>>>(d_tensor, elements);
    check_cuda(cudaGetLastError(), "launch relu_kernel");
}

inline void apply_dropout(float* d_tensor, int elements, float drop_prob, cudaStream_t stream) {
    (void)d_tensor;
    (void)elements;
    (void)drop_prob;
    (void)stream;
}

__global__ void softmax_cross_entropy_kernel(const float* d_logits,
                                             const int* d_labels,
                                             int num_nodes,
                                             int num_classes,
                                             float* d_loss) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float total_loss = 0.0f;
    for (int i = 0; i < num_nodes; ++i) {
        const float* row = d_logits + static_cast<std::size_t>(i) * num_classes;

        float max_logit = row[0];
        for (int c = 1; c < num_classes; ++c) {
            max_logit = fmaxf(max_logit, row[c]);
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            sum_exp += expf(row[c] - max_logit);
        }

        const int y = d_labels[i];
        const float log_prob = (row[y] - max_logit) - logf(sum_exp);
        total_loss += -log_prob;
    }

    *d_loss = total_loss / static_cast<float>(num_nodes);
}

inline void softmax_cross_entropy(const float* d_logits,
                                  const int* d_labels,
                                  int num_nodes,
                                  int num_classes,
                                  float* d_loss) {
    softmax_cross_entropy_kernel<<<1, 1>>>(d_logits, d_labels, num_nodes, num_classes, d_loss);
    check_cuda(cudaGetLastError(), "launch softmax_cross_entropy_kernel");
}