#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

#include "gcn_layers.cuh"

struct Options {
    std::string graph_prefix = "data/cora";  // expects .csr, .feat, .label
    int hidden_dim = 128;
    int layers = 2;
    std::string impl = "baseline";  // baseline | fused
    bool verify = true;
    std::string dump_path = "";
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--graph") == 0 || strcmp(argv[i], "-g") == 0) && i + 1 < argc) {
            opt.graph_prefix = argv[++i];
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            opt.hidden_dim = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            opt.layers = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            opt.impl = argv[++i];
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            opt.dump_path = argv[++i];
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opt.verify = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout
                << "Usage: ./dgcn --graph data/cora --hidden 128 --layers 2 --impl baseline \\\n"
                << "  [--dump outputs.bin] [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
        }
    }

    if (opt.hidden_dim <= 0 || opt.layers < 1) {
        throw std::invalid_argument("hidden and layers must be positive");
    }
    if (opt.layers != 2) {
        std::cerr << "Warning: this main.cu currently implements exactly 2 GCN layers; "
                  << "ignoring --layers=" << opt.layers << " and proceeding with 2 layers.\n";
    }
    return opt;
}

static void init_weights(std::vector<float>& w, int fan_in, int fan_out, unsigned seed) {
    std::mt19937 rng(seed);
    const float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (float& x : w) x = dist(rng);
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    GraphData graph;
    build_graph_from_files(opt.graph_prefix, graph);

    cusparseHandle_t cusparse;
    check_cusparse(cusparseCreate(&cusparse), "cusparseCreate");

    cublasHandle_t cublas;
    check_cublas(cublasCreate(&cublas), "cublasCreate");

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    check_cublas(cublasSetStream(cublas, stream), "cublasSetStream");
    check_cusparse(cusparseSetStream(cusparse, stream), "cusparseSetStream");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    DeviceGCNWorkspace workspace;
    allocate_device_graph(graph, workspace);

    const int N = graph.num_nodes;
    const int Fin = graph.feature_dim;
    const int H = opt.hidden_dim;
    const int C = graph.num_classes;

    // Buffers for DGL-matching forward:
    // pre1   = X W0        [N, H]
    // post1  = A_hat pre1  [N, H]
    // pre2   = post1 W1    [N, C]
    // logits = A_hat pre2  [N, C]
    float* d_pre1 = nullptr;
    float* d_post1 = nullptr;
    float* d_pre2 = nullptr;
    float* d_w0 = nullptr;   // [Fin, H]
    float* d_w1 = nullptr;   // [H, C]

    check_cuda(cudaMalloc(&d_pre1, static_cast<size_t>(N) * H * sizeof(float)),
               "cudaMalloc d_pre1");
    check_cuda(cudaMalloc(&d_post1, static_cast<size_t>(N) * H * sizeof(float)),
               "cudaMalloc d_post1");
    check_cuda(cudaMalloc(&d_pre2, static_cast<size_t>(N) * C * sizeof(float)),
               "cudaMalloc d_pre2");
    check_cuda(cudaMalloc(&d_w0, static_cast<size_t>(Fin) * H * sizeof(float)),
               "cudaMalloc d_w0");
    check_cuda(cudaMalloc(&d_w1, static_cast<size_t>(H) * C * sizeof(float)),
               "cudaMalloc d_w1");

    std::vector<float> h_w0(static_cast<size_t>(Fin) * H);
    std::vector<float> h_w1(static_cast<size_t>(H) * C);
    init_weights(h_w0, Fin, H, 1234u);
    init_weights(h_w1, H, C, 5678u);

    check_cuda(cudaMemcpy(d_w0, h_w0.data(),
                          h_w0.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "copy W0 to device");
    check_cuda(cudaMemcpy(d_w1, h_w1.data(),
                          h_w1.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "copy W1 to device");

    // Keep only weights in weights.bin, matching compare_with_dgl.py
    std::vector<float> h_weights;
    h_weights.reserve(h_w0.size() + h_w1.size());
    h_weights.insert(h_weights.end(), h_w0.begin(), h_w0.end());
    h_weights.insert(h_weights.end(), h_w1.begin(), h_w1.end());

    if (!opt.dump_path.empty()) {
        std::ofstream wofs("weights.bin", std::ios::binary);
        if (!wofs) {
            throw std::runtime_error("Could not open weights.bin for writing");
        }
        wofs.write(reinterpret_cast<const char*>(h_weights.data()),
                   static_cast<std::streamsize>(h_weights.size() * sizeof(float)));
        wofs.close();
        std::cout << "Weights dumped to weights.bin\n";
    }

    float elapsed_ms = 0.0f;

    if (opt.impl == "baseline" || opt.impl == "fused") {
        check_cuda(cudaEventRecord(start, stream), "record start");

        // Layer 1: X * W0   -> [N, H]
        run_dense_layer(
            cublas,
            N,      // M
            Fin,    // K
            H,      // N
            workspace.d_features_in,
            d_w0,
            d_pre1
        );

        // Layer 1: A_hat * (X * W0) -> [N, H]
        run_sparse_dense_mm(
            cusparse,
            workspace,
            N,      // rows
            H,      // cols
            H,      // unused by helper, pass H
            d_pre1,
            d_post1
        );

        apply_activation(d_post1, N * H, stream);

        // Layer 2: hidden * W1 -> [N, C]
        run_dense_layer(
            cublas,
            N,
            H,
            C,
            d_post1,
            d_w1,
            d_pre2
        );

        // Layer 2: A_hat * (hidden * W1) -> logits [N, C]
        run_sparse_dense_mm(
            cusparse,
            workspace,
            N,
            C,
            C,
            d_pre2,
            workspace.d_logits
        );

        check_cuda(cudaEventRecord(stop, stream), "record stop");
        check_cuda(cudaEventSynchronize(stop), "sync stop");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");
    } else {
        throw std::invalid_argument("Unknown --impl=" + opt.impl);
    }

    std::vector<float> h_logits(static_cast<size_t>(N) * C, 0.0f);
    check_cuda(cudaMemcpy(h_logits.data(),
                          workspace.d_logits,
                          h_logits.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy logits back to host");

    if (!opt.dump_path.empty()) {
        std::ofstream ofs(opt.dump_path, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Failed to open dump path: " + opt.dump_path);
        }
        ofs.write(reinterpret_cast<const char*>(h_logits.data()),
                  static_cast<std::streamsize>(h_logits.size() * sizeof(float)));
        ofs.close();
        std::cout << "Logits dumped to " << opt.dump_path << "\n";
    }

    if (opt.verify) {
        std::cout << "Verification hook TODO: run compare_with_dgl.py separately.\n";
    }

    if (elapsed_ms > 0.0f) {
        std::cout << std::fixed << std::setprecision(2)
                  << "Impl=" << opt.impl
                  << " Graph=" << opt.graph_prefix
                  << " Hidden=" << opt.hidden_dim
                  << " Layers=2"
                  << " Time(ms)=" << elapsed_ms
                  << " Edges/s=" << (graph.nnz / (elapsed_ms * 1e-3f))
                  << '\n';
    } else {
        std::cout << "Forward pass executed.\n";
    }

    cudaFree(d_pre1);
    cudaFree(d_post1);
    cudaFree(d_pre2);
    cudaFree(d_w0);
    cudaFree(d_w1);

    destroy_device_graph(workspace);

    check_cuda(cudaEventDestroy(start), "destroy start event");
    check_cuda(cudaEventDestroy(stop), "destroy stop event");
    check_cuda(cudaStreamDestroy(stream), "destroy stream");
    check_cublas(cublasDestroy(cublas), "cublasDestroy");
    check_cusparse(cusparseDestroy(cusparse), "cusparseDestroy");

    return 0;
}