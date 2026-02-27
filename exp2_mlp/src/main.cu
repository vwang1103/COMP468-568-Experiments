#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include "mlp_layers.cuh"

struct Options {
    std::vector<int> layers = {1024, 2048, 1024};  // includes input dim and final output dim
    int batch = 128;
    std::string activation = "relu";
    std::string impl = "baseline";  // baseline | activation_fused
    bool verify = true;
};

std::vector<int> parse_layers_list(const std::string& csv) {
    std::vector<int> dims;
    size_t start = 0;
    while (start < csv.size()) {
        size_t comma = csv.find(',', start);
        const size_t len = (comma == std::string::npos) ? (csv.size() - start) : (comma - start);
        if (len > 0) {
            dims.push_back(std::stoi(csv.substr(start, len)));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return dims;
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            opt.layers = parse_layers_list(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            opt.batch = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) {
            opt.activation = argv[++i];
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            opt.impl = argv[++i];
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opt.verify = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ./dmlp --layers 1024,2048,1024 --batch 128 --activation relu \\\n  --impl baseline|activation_fused [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
        }
    }
    if (opt.layers.size() < 2) {
        throw std::invalid_argument("--layers must contain at least two integers (input/output)");
    }
    return opt;
}

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(err));
    }
}

void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuBLAS error");
    }
}

void seed_tensor(std::vector<float>& data, float scale) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = scale * std::sin(0.11f * static_cast<float>(i));
    }
}

// CPU reference
// weights layout: [out][in] row-major
// forward per layer: Y = act( X * W^T + b )
static inline float relu_cpu(float x) { return (x > 0.0f) ? x : 0.0f; }

static inline float gelu_cpu(float x) {
    // tanh GELU approximation (matches the common GPU approx)
    constexpr float k0 = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float k1 = 0.044715f;
    float x3 = x * x * x;
    float t = k0 * (x + k1 * x3);
    return 0.5f * x * (1.0f + std::tanh(t));
}

static inline float apply_activation_cpu(const std::string& act, float x) {
    if (act == "relu") return relu_cpu(x);
    if (act == "gelu") return gelu_cpu(x);
    return x; // unknown => no-op
}

void mlp_cpu_reference(const std::vector<int>& layers,
                       int batch,
                       const std::vector<float>& weights,
                       const std::vector<float>& biases,
                       const std::vector<size_t>& weight_offsets,
                       const std::vector<size_t>& bias_offsets,
                       const std::vector<float>& input,
                       std::vector<float>& output,
                       const std::string& activation) {
    const int num_layers = static_cast<int>(layers.size()) - 1;

    // Ping-pong buffers for activations on CPU
    std::vector<float> a = input; // [batch][layers[0]]
    std::vector<float> b;

    for (int layer = 0; layer < num_layers; ++layer) {
        const int in_dim  = layers[layer];
        const int out_dim = layers[layer + 1];

        b.assign(static_cast<size_t>(batch) * out_dim, 0.0f);

        const float* W = weights.data() + weight_offsets[layer]; // [out_dim][in_dim]
        const float* bias = biases.data() + bias_offsets[layer]; // [out_dim]

        for (int r = 0; r < batch; ++r) {
            const float* a_row = a.data() + static_cast<size_t>(r) * in_dim;
            float* b_row       = b.data() + static_cast<size_t>(r) * out_dim;

            for (int j = 0; j < out_dim; ++j) {
                double acc = 0.0;
                const float* Wrow = W + (size_t)j * in_dim; // W[j,:]
                for (int i = 0; i < in_dim; ++i) {
                    acc += static_cast<double>(a_row[i]) * static_cast<double>(Wrow[i]);
                }
                acc += static_cast<double>(bias[j]);
                float out = static_cast<float>(acc);
                out = apply_activation_cpu(activation, out);
                b_row[j] = out;
            }
        }

        a.swap(b);
    }

    output = a; // final [batch][layers.back()]
}

// -----------------------
// Main
// -----------------------
int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    const int batch = opt.batch;
    const int L = static_cast<int>(opt.layers.size()) - 1;

    // Offsets for packed weights/biases
    std::vector<size_t> weight_offsets(L, 0);
    std::vector<size_t> bias_offsets(L, 0);

    size_t weight_cursor = 0;
    size_t bias_cursor = 0;
    int max_dim = 0;
    for (int d : opt.layers) max_dim = std::max(max_dim, d);

    for (int layer = 0; layer < L; ++layer) {
        const int in  = opt.layers[layer];
        const int out = opt.layers[layer + 1];
        weight_offsets[layer] = weight_cursor;
        bias_offsets[layer] = bias_cursor;

        // weights layout: [out][in]
        weight_cursor += static_cast<size_t>(out) * in;
        bias_cursor += static_cast<size_t>(out);
    }

    const size_t input_elems  = static_cast<size_t>(batch) * opt.layers.front();
    const size_t output_elems = static_cast<size_t>(batch) * opt.layers.back();
    const size_t workspace_elems = static_cast<size_t>(batch) * static_cast<size_t>(max_dim);

    // Host buffers
    std::vector<float> h_input(input_elems);
    std::vector<float> h_weights(weight_cursor);
    std::vector<float> h_biases(bias_cursor);
    std::vector<float> h_output(output_elems, 0.0f);
    std::vector<float> h_ref(output_elems, 0.0f);

    seed_tensor(h_input, 1.0f);
    seed_tensor(h_weights, 0.25f);
    seed_tensor(h_biases, 0.01f);

    // Device buffers
    float* d_input = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_weights = nullptr;
    float* d_biases = nullptr;

    check_cuda(cudaMalloc(&d_input,   input_elems * sizeof(float)), "cudaMalloc d_input");
    check_cuda(cudaMalloc(&d_a,       workspace_elems * sizeof(float)), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_b,       workspace_elems * sizeof(float)), "cudaMalloc d_b");
    check_cuda(cudaMalloc(&d_weights, weight_cursor * sizeof(float)), "cudaMalloc d_weights");
    check_cuda(cudaMalloc(&d_biases,  bias_cursor * sizeof(float)), "cudaMalloc d_biases");

    check_cuda(cudaMemcpy(d_input,   h_input.data(),   input_elems * sizeof(float), cudaMemcpyHostToDevice),
               "H2D input");
    check_cuda(cudaMemcpy(d_weights, h_weights.data(), weight_cursor * sizeof(float), cudaMemcpyHostToDevice),
               "H2D weights");
    check_cuda(cudaMemcpy(d_biases,  h_biases.data(),  bias_cursor * sizeof(float), cudaMemcpyHostToDevice),
               "H2D biases");

    // Stream + cuBLAS + events
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "create stream");

    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "cublasCreate");
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    // Stable timing parameters
    const int WARMUP_ITERS = 20;
    const int TIMED_ITERS  = 200;

    // One forward pass helper (end-to-end) – restores input each iter so work is identical
    auto forward_once = [&]() {
        // reset input into d_a
        check_cuda(cudaMemcpyAsync(d_a, d_input, input_elems * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream),
                   "iter restore input");

        for (int layer = 0; layer < L; ++layer) {
            LayerShape shape{batch, opt.layers[layer], opt.layers[layer + 1]};
            const float* d_w = d_weights + weight_offsets[layer];
            const float* d_bias = d_biases + bias_offsets[layer];

            run_gemm_layer(d_a, d_w, d_b, shape, handle);

            if (opt.impl == "activation_fused") {
                launch_fused_bias_activation(d_bias, opt.activation, d_b, shape, stream);
            } else {
                launch_bias_add(d_bias, d_b, shape, stream);
                launch_activation(opt.activation, d_b, shape, stream);
            }

            std::swap(d_a, d_b);
        }
    };

    // Warmup
    for (int t = 0; t < WARMUP_ITERS; ++t) {
        forward_once();
    }
    check_cuda(cudaStreamSynchronize(stream), "warmup sync");

    // Timed loop
    check_cuda(cudaEventRecord(start, stream), "record start");
    for (int t = 0; t < TIMED_ITERS; ++t) {
        forward_once();
    }
    check_cuda(cudaEventRecord(stop, stream), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");

    float total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "elapsed time");
    float elapsed_ms = total_ms / static_cast<float>(TIMED_ITERS);

    // After the last forward_once(), final output is in d_a
    check_cuda(cudaMemcpyAsync(h_output.data(), d_a, output_elems * sizeof(float),
                               cudaMemcpyDeviceToHost, stream),
               "D2H output");
    check_cuda(cudaStreamSynchronize(stream), "sync D2H output");

    // Verify (optional; expensive for large sizes but OK if opt.verify is true)
    if (opt.verify) {
        mlp_cpu_reference(opt.layers,
                          batch,
                          h_weights,
                          h_biases,
                          weight_offsets,
                          bias_offsets,
                          h_input,
                          h_ref,
                          opt.activation);

        float max_abs = 0.0f;
        float max_rel = 0.0f;
        for (size_t i = 0; i < output_elems; ++i) {
            float a = h_output[i];
            float b = h_ref[i];
            float diff = std::fabs(a - b);
            max_abs = std::max(max_abs, diff);
            float denom = std::max(1.0f, std::fabs(b));
            max_rel = std::max(max_rel, diff / denom);
        }

        std::cout << std::scientific;
        std::cout << "Verify: max_abs_diff=" << max_abs << " max_rel_diff=" << max_rel << "\n";
    }

    // Report perf (mean time per forward pass)
    if (elapsed_ms > 0.0f) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Impl=" << opt.impl << " Batch=" << batch << " Layers=";
        for (size_t i = 0; i < opt.layers.size(); ++i) {
            std::cout << opt.layers[i];
            if (i + 1 < opt.layers.size()) std::cout << "x";
        }
        std::cout << " Time(ms)=" << elapsed_ms
                  << " GFLOP/s=" << mlp_gflops(opt.layers, batch, elapsed_ms) << "\n";
    }

    // Cleanup
    check_cuda(cudaEventDestroy(start), "destroy start");
    check_cuda(cudaEventDestroy(stop), "destroy stop");
    check_cublas(cublasDestroy(handle), "cublasDestroy");
    check_cuda(cudaStreamDestroy(stream), "destroy stream");

    check_cuda(cudaFree(d_input), "cudaFree d_input");
    check_cuda(cudaFree(d_a), "cudaFree d_a");
    check_cuda(cudaFree(d_b), "cudaFree d_b");
    check_cuda(cudaFree(d_weights), "cudaFree d_weights");
    check_cuda(cudaFree(d_biases), "cudaFree d_biases");

    return 0;
}