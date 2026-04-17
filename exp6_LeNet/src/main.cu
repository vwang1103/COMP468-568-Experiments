#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lenet_layers.cuh"

struct Options {
    int batch = 32;
    std::string algo = "implicit_gemm";   // cuDNN conv algo hint
    std::string impl = "baseline";        // baseline | fused
    bool verify = true;
    std::string dump_path = "";           // optional binary file for logits
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--batch") == 0 || strcmp(argv[i], "-b") == 0) && i + 1 < argc) {
            opt.batch = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--algo") == 0 && i + 1 < argc) {
            opt.algo = argv[++i];
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            opt.impl = argv[++i];
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            opt.dump_path = argv[++i];
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opt.verify = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout
                << "Usage: ./dlenet --batch N "
                << "--algo implicit_gemm|implicit_precomp|fft "
                << "--impl baseline|fused "
                << "[--dump outputs.bin] [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
        }
    }

    if (opt.batch <= 0) {
        throw std::invalid_argument("Batch must be > 0");
    }
    if (opt.impl != "baseline" && opt.impl != "fused") {
        throw std::invalid_argument("Unknown --impl=" + opt.impl);
    }

    return opt;
}

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(err));
    }
}

void check_cudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : " + cudnnGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuBLAS error");
    }
}

void seed_tensor(std::vector<float>& vec, float scale) {
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = scale * std::sin(0.017f * static_cast<float>(i));
    }
}

void lenet_cpu_reference(const Options& opt,
                         const LenetShape& shape,
                         const std::vector<float>& weights,
                         const std::vector<size_t>& weight_offsets,
                         const std::vector<float>& biases,
                         const std::vector<size_t>& bias_offsets,
                         const std::vector<float>& input,
                         std::vector<float>& output) {
    (void)opt;
    const int B = shape.batch;
    const int conv1_h = 28, conv1_w = 28;
    const int pool1_h = 14, pool1_w = 14;
    const int conv2_h = 10, conv2_w = 10;
    const int pool2_h = 5,  pool2_w = 5;
    const int flat = 16 * pool2_h * pool2_w;

    std::vector<float> conv1_out(B * 6 * conv1_h * conv1_w, 0.0f);
    std::vector<float> pool1_out(B * 6 * pool1_h * pool1_w, 0.0f);
    std::vector<float> conv2_out(B * 16 * conv2_h * conv2_w, 0.0f);
    std::vector<float> pool2_out(B * 16 * pool2_h * pool2_w, 0.0f);
    std::vector<float> fc1_out(B * 120, 0.0f);
    std::vector<float> fc2_out(B * 84, 0.0f);

    auto idx4 = [](int n, int c, int h, int w, int C, int H, int W) {
        return ((n * C + c) * H + h) * W + w;
    };

    // Conv1: (B,1,32,32) -> (B,6,28,28), cross-correlation + bias + tanh
    const float* w1 = &weights[weight_offsets[0]];
    const float* b1 = &biases[bias_offsets[0]];
    for (int n = 0; n < B; n++) {
        for (int k = 0; k < 6; k++) {
            for (int oh = 0; oh < conv1_h; oh++) {
                for (int ow = 0; ow < conv1_w; ow++) {
                    float val = b1[k];
                    for (int c = 0; c < 1; c++) {
                        for (int kh = 0; kh < 5; kh++) {
                            for (int kw = 0; kw < 5; kw++) {
                                val += input[idx4(n, c, oh + kh, ow + kw, 1, 32, 32)]
                                     * w1[idx4(k, c, kh, kw, 1, 5, 5)];
                            }
                        }
                    }
                    conv1_out[idx4(n, k, oh, ow, 6, conv1_h, conv1_w)] = std::tanh(val);
                }
            }
        }
    }

    // Pool1: max 2x2 stride 2
    for (int n = 0; n < B; n++) {
        for (int c = 0; c < 6; c++) {
            for (int oh = 0; oh < pool1_h; oh++) {
                for (int ow = 0; ow < pool1_w; ow++) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < 2; kh++) {
                        for (int kw = 0; kw < 2; kw++) {
                            sum += conv1_out[idx4(n, c, oh * 2 + kh, ow * 2 + kw, 6, conv1_h, conv1_w)];
                        }
                    }
                    pool1_out[idx4(n, c, oh, ow, 6, pool1_h, pool1_w)] = sum * 0.25f;
                }
            }
        }
    }

    // Conv2: (B,6,14,14) -> (B,16,10,10), cross-correlation + bias + tanh
    const float* w2 = &weights[weight_offsets[1]];
    const float* b2 = &biases[bias_offsets[1]];
    for (int n = 0; n < B; n++) {
        for (int k = 0; k < 16; k++) {
            for (int oh = 0; oh < conv2_h; oh++) {
                for (int ow = 0; ow < conv2_w; ow++) {
                    float val = b2[k];
                    for (int c = 0; c < 6; c++) {
                        for (int kh = 0; kh < 5; kh++) {
                            for (int kw = 0; kw < 5; kw++) {
                                val += pool1_out[idx4(n, c, oh + kh, ow + kw, 6, pool1_h, pool1_w)]
                                     * w2[idx4(k, c, kh, kw, 6, 5, 5)];
                            }
                        }
                    }
                    conv2_out[idx4(n, k, oh, ow, 16, conv2_h, conv2_w)] = std::tanh(val);
                }
            }
        }
    }

    // Pool2: max 2x2 stride 2
    for (int n = 0; n < B; n++) {
        for (int c = 0; c < 16; c++) {
            for (int oh = 0; oh < pool2_h; oh++) {
                for (int ow = 0; ow < pool2_w; ow++) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < 2; kh++) {
                        for (int kw = 0; kw < 2; kw++) {
                            sum += conv2_out[idx4(n, c, oh * 2 + kh, ow * 2 + kw, 16, conv2_h, conv2_w)];
                        }
                    }
                    pool2_out[idx4(n, c, oh, ow, 16, pool2_h, pool2_w)] = sum * 0.25f;
                }
            }
        }
    }

    // FC1: (B,400) -> (B,120) + bias + tanh
    const float* wfc1 = &weights[weight_offsets[2]];
    const float* bfc1 = &biases[bias_offsets[2]];
    for (int n = 0; n < B; n++) {
        for (int j = 0; j < 120; j++) {
            float val = bfc1[j];
            for (int i = 0; i < flat; i++) {
                val += pool2_out[n * flat + i] * wfc1[j * flat + i];
            }
            fc1_out[n * 120 + j] = std::tanh(val);
        }
    }

    // FC2: (B,120) -> (B,84) + bias + tanh
    const float* wfc2 = &weights[weight_offsets[3]];
    const float* bfc2 = &biases[bias_offsets[3]];
    for (int n = 0; n < B; n++) {
        for (int j = 0; j < 84; j++) {
            float val = bfc2[j];
            for (int i = 0; i < 120; i++) {
                val += fc1_out[n * 120 + i] * wfc2[j * 120 + i];
            }
            fc2_out[n * 84 + j] = std::tanh(val);
        }
    }

    // FC3: (B,84) -> (B,10) + bias, no activation
    const float* wfc3 = &weights[weight_offsets[4]];
    const float* bfc3 = &biases[bias_offsets[4]];
    for (int n = 0; n < B; n++) {
        for (int j = 0; j < 10; j++) {
            float val = bfc3[j];
            for (int i = 0; i < 84; i++) {
                val += fc2_out[n * 84 + i] * wfc3[j * 84 + i];
            }
            output[n * 10 + j] = val;
        }
    }
}

double conv_only_gflops(int B, int Cout, int H, int W, int Cin, int K, double ms) {
    double flops = 2.0 * B * Cout * H * W * Cin * K * K;
    return flops / (ms * 1e6);
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    LenetShape shape = make_lenet_shape(opt.batch);

    std::vector<float> h_input(shape.input_elements);
    std::vector<float> h_weights(shape.total_weight_elements);
    std::vector<float> h_biases(shape.total_bias_elements);
    std::vector<float> h_output(shape.output_elements, 0.0f);
    std::vector<float> h_ref(shape.output_elements, 0.0f);

    seed_tensor(h_input, 1.0f);
    seed_tensor(h_weights, 0.05f);
    seed_tensor(h_biases, 0.01f);

    float* d_input = nullptr;
    void*  d_workspace = nullptr;
    float* d_conv1_out = nullptr;
    float* d_conv2_out = nullptr;
    float* d_pool1_out = nullptr;
    float* d_pool2_out = nullptr;
    float* d_fc1_out = nullptr;
    float* d_fc2_out = nullptr;
    float* d_fc3_out = nullptr;
    float* d_weights = nullptr;
    float* d_biases = nullptr;

    check_cuda(cudaMalloc(&d_input,     shape.input_elements         * sizeof(float)), "alloc input");
    check_cuda(cudaMalloc(&d_conv1_out, shape.conv1_out_elems        * sizeof(float)), "alloc conv1_out");
    check_cuda(cudaMalloc(&d_pool1_out, shape.pool1_out_elems        * sizeof(float)), "alloc pool1_out");
    check_cuda(cudaMalloc(&d_conv2_out, shape.conv2_out_elems        * sizeof(float)), "alloc conv2_out");
    check_cuda(cudaMalloc(&d_pool2_out, shape.pool2_out_elems        * sizeof(float)), "alloc pool2_out");
    check_cuda(cudaMalloc(&d_fc1_out,   shape.fc1_out_elems          * sizeof(float)), "alloc fc1_out");
    check_cuda(cudaMalloc(&d_fc2_out,   shape.fc2_out_elems          * sizeof(float)), "alloc fc2_out");
    check_cuda(cudaMalloc(&d_fc3_out,   shape.output_elements        * sizeof(float)), "alloc fc3_out");
    check_cuda(cudaMalloc(&d_weights,   shape.total_weight_elements  * sizeof(float)), "alloc weights");
    check_cuda(cudaMalloc(&d_biases,    shape.total_bias_elements    * sizeof(float)), "alloc biases");

    check_cuda(cudaMemcpy(d_input,   h_input.data(),   shape.input_elements        * sizeof(float), cudaMemcpyHostToDevice), "copy input");
    check_cuda(cudaMemcpy(d_weights, h_weights.data(), shape.total_weight_elements * sizeof(float), cudaMemcpyHostToDevice), "copy weights");
    check_cuda(cudaMemcpy(d_biases,  h_biases.data(),  shape.total_bias_elements   * sizeof(float), cudaMemcpyHostToDevice), "copy biases");

    cudnnHandle_t cudnn;
    check_cudnn(cudnnCreate(&cudnn), "cudnnCreate");

    cublasHandle_t cublas;
    check_cublas(cublasCreate(&cublas), "cublasCreate");

    LenetDescriptors descs;
    create_lenet_descriptors(shape, descs);

    cudnnConvolutionFwdAlgo_t fwd_algo = parse_algo(opt.algo);

    // Query workspace for plain conv path and use max(conv1, conv2)
    size_t ws1 = query_conv_workspace(cudnn, shape, descs, fwd_algo, false);
    size_t ws2 = query_conv_workspace(cudnn, shape, descs, fwd_algo, true);
    size_t ws_required = (ws1 > ws2) ? ws1 : ws2;

    // Fused conv+bias path may need more workspace than plain conv for some algos.
    // Query it too when --impl=fused.
    cudnnTensorDescriptor_t conv1_bias_desc = nullptr;
    cudnnTensorDescriptor_t conv2_bias_desc = nullptr;

    check_cudnn(cudnnCreateTensorDescriptor(&conv1_bias_desc), "create conv1_bias_desc");
    check_cudnn(
        cudnnSetTensor4dDescriptor(conv1_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, LenetShape::conv1_out_channels, 1, 1),
        "set conv1_bias_desc");

    check_cudnn(cudnnCreateTensorDescriptor(&conv2_bias_desc), "create conv2_bias_desc");
    check_cudnn(
        cudnnSetTensor4dDescriptor(conv2_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, LenetShape::conv2_out_channels, 1, 1),
        "set conv2_bias_desc");

    cudnnActivationDescriptor_t identity_activation = nullptr;
    check_cudnn(cudnnCreateActivationDescriptor(&identity_activation), "create identity activation");
    check_cudnn(
        cudnnSetActivationDescriptor(identity_activation,
                                     CUDNN_ACTIVATION_IDENTITY,
                                     CUDNN_PROPAGATE_NAN,
                                     0.0),
        "set identity activation");

    if (opt.impl == "fused") {
        size_t fused_ws1 = 0;
        size_t fused_ws2 = 0;

        // Conv1 fused workspace
        check_cudnn(
            cudnnGetConvolutionForwardWorkspaceSize(
                cudnn,
                descs.input_desc,
                descs.conv1_filter,
                descs.conv1_desc,
                descs.conv1_out_desc,
                fwd_algo,
                &fused_ws1),
            "query fused conv1 workspace");

        // Conv2 fused workspace
        check_cudnn(
            cudnnGetConvolutionForwardWorkspaceSize(
                cudnn,
                descs.pool1_out_desc,
                descs.conv2_filter,
                descs.conv2_desc,
                descs.conv2_out_desc,
                fwd_algo,
                &fused_ws2),
            "query fused conv2 workspace");

        size_t fused_required = (fused_ws1 > fused_ws2) ? fused_ws1 : fused_ws2;
        if (fused_required > ws_required) ws_required = fused_required;
    }

    size_t ws_alloc = (ws_required == 0) ? 1 : ws_required;
    check_cuda(cudaMalloc(&d_workspace, ws_alloc), "alloc workspace");

    /* ---------------- conv-only cuDNN benchmark ---------------- */
    {
        const float alpha = 1.0f, beta = 0.0f;
        constexpr int CONV_WARMUP = 20;
        constexpr int CONV_ITERS = 200;

        cudaEvent_t cs, ce;
        check_cuda(cudaEventCreate(&cs), "conv start");
        check_cuda(cudaEventCreate(&ce), "conv stop");

        // Warmup
        for (int i = 0; i < CONV_WARMUP; i++) {
            cudnnConvolutionForward(
                cudnn, &alpha,
                descs.input_desc, d_input,
                descs.conv1_filter, d_weights + shape.weight_offsets[0],
                descs.conv1_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                d_workspace, ws_alloc,
                &beta,
                descs.conv1_out_desc, d_conv1_out);
        }
        check_cuda(cudaDeviceSynchronize(), "conv warmup");

        // Timed
        check_cuda(cudaEventRecord(cs), "conv start");
        for (int i = 0; i < CONV_ITERS; i++) {
            cudnnConvolutionForward(
                cudnn, &alpha,
                descs.input_desc, d_input,
                descs.conv1_filter, d_weights + shape.weight_offsets[0],
                descs.conv1_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                d_workspace, ws_alloc,
                &beta,
                descs.conv1_out_desc, d_conv1_out);
        }
        check_cuda(cudaEventRecord(ce), "conv stop");
        check_cuda(cudaEventSynchronize(ce), "conv sync");

        float conv_ms;
        check_cuda(cudaEventElapsedTime(&conv_ms, cs, ce), "conv elapsed");
        conv_ms /= CONV_ITERS;

        double conv_gflops = conv_only_gflops(
            shape.batch,
            LenetShape::conv1_out_channels,
            28, 28,
            LenetShape::in_channels,
            LenetShape::conv1_kernel,
            conv_ms);

        std::cout << std::fixed << std::setprecision(2)
                  << "cuDNN implicit_gemm conv-only: "
                  << "Time(ms)=" << conv_ms
                  << " GFLOP/s=" << conv_gflops
                  << std::endl;

        cudaEventDestroy(ce);
        cudaEventDestroy(cs);
    }

    /* --------- existing forward execution (baseline/fused) --------- */

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start");
    check_cuda(cudaEventCreate(&stop), "create stop");

    auto run_baseline = [&]() {
        // Conv1 -> bias -> tanh
        run_lenet_conv(cudnn, shape, descs, d_input,
                       d_weights + shape.weight_offsets[0], d_conv1_out,
                       d_workspace, ws_alloc, opt.algo, false);
        {
            const float a = 1.0f, b = 1.0f;
            check_cudnn(
                cudnnAddTensor(cudnn, &a, conv1_bias_desc,
                               d_biases + shape.bias_offsets[0],
                               &b,
                               descs.conv1_out_desc, d_conv1_out),
                "conv1 bias");
        }
        {
            const float a = 1.0f, b = 0.0f;
            check_cudnn(
                cudnnActivationForward(cudnn, descs.activation,
                                       &a,
                                       descs.conv1_out_desc, d_conv1_out,
                                       &b,
                                       descs.conv1_out_desc, d_conv1_out),
                "conv1 tanh");
        }

        // Pool1
        run_lenet_pool(cudnn, descs, d_conv1_out, d_pool1_out, false);

        // Conv2 -> bias -> tanh
        run_lenet_conv(cudnn, shape, descs, d_pool1_out,
                       d_weights + shape.weight_offsets[1], d_conv2_out,
                       d_workspace, ws_alloc, opt.algo, true);
        {
            const float a = 1.0f, b = 1.0f;
            check_cudnn(
                cudnnAddTensor(cudnn, &a, conv2_bias_desc,
                               d_biases + shape.bias_offsets[1],
                               &b,
                               descs.conv2_out_desc, d_conv2_out),
                "conv2 bias");
        }
        {
            const float a = 1.0f, b = 0.0f;
            check_cudnn(
                cudnnActivationForward(cudnn, descs.activation,
                                       &a,
                                       descs.conv2_out_desc, d_conv2_out,
                                       &b,
                                       descs.conv2_out_desc, d_conv2_out),
                "conv2 tanh");
        }

        // Pool2
        run_lenet_pool(cudnn, descs, d_conv2_out, d_pool2_out, true);

        // FC layers
        run_fc_layer(cublas, shape, 0, d_pool2_out,
                     d_weights + shape.weight_offsets[2],
                     d_biases  + shape.bias_offsets[2],
                     d_fc1_out, nullptr);

        run_fc_layer(cublas, shape, 1, d_fc1_out,
                     d_weights + shape.weight_offsets[3],
                     d_biases  + shape.bias_offsets[3],
                     d_fc2_out, nullptr);

        run_fc_layer(cublas, shape, 2, d_fc2_out,
                     d_weights + shape.weight_offsets[4],
                     d_biases  + shape.bias_offsets[4],
                     d_fc3_out, nullptr);
    };

    auto run_fused = [&]() {
        const float one = 1.0f;
        const float zero = 0.0f;

        // Conv1 + bias fused, activation bypassed as IDENTITY
        check_cudnn(
            cudnnConvolutionBiasActivationForward(
                cudnn,
                &one,
                descs.input_desc, d_input,
                descs.conv1_filter, d_weights + shape.weight_offsets[0],
                descs.conv1_desc,
                fwd_algo,
                d_workspace, ws_alloc,
                &zero,
                descs.conv1_out_desc, d_conv1_out,
                conv1_bias_desc, d_biases + shape.bias_offsets[0],
                identity_activation,
                descs.conv1_out_desc, d_conv1_out),
            "fused conv1");

        // Separate tanh
        check_cudnn(
            cudnnActivationForward(
                cudnn,
                descs.activation,
                &one,
                descs.conv1_out_desc, d_conv1_out,
                &zero,
                descs.conv1_out_desc, d_conv1_out),
            "conv1 tanh");

        // Pool1
        run_lenet_pool(cudnn, descs, d_conv1_out, d_pool1_out, false);

        // Conv2 + bias fused, activation bypassed as IDENTITY
        check_cudnn(
            cudnnConvolutionBiasActivationForward(
                cudnn,
                &one,
                descs.pool1_out_desc, d_pool1_out,
                descs.conv2_filter, d_weights + shape.weight_offsets[1],
                descs.conv2_desc,
                fwd_algo,
                d_workspace, ws_alloc,
                &zero,
                descs.conv2_out_desc, d_conv2_out,
                conv2_bias_desc, d_biases + shape.bias_offsets[1],
                identity_activation,
                descs.conv2_out_desc, d_conv2_out),
            "fused conv2");

        // Separate tanh
        check_cudnn(
            cudnnActivationForward(
                cudnn,
                descs.activation,
                &one,
                descs.conv2_out_desc, d_conv2_out,
                &zero,
                descs.conv2_out_desc, d_conv2_out),
            "conv2 tanh");

        // Pool2
        run_lenet_pool(cudnn, descs, d_conv2_out, d_pool2_out, true);

        // FC layers unchanged
        run_fc_layer(cublas, shape, 0, d_pool2_out,
                     d_weights + shape.weight_offsets[2],
                     d_biases  + shape.bias_offsets[2],
                     d_fc1_out, nullptr);

        run_fc_layer(cublas, shape, 1, d_fc1_out,
                     d_weights + shape.weight_offsets[3],
                     d_biases  + shape.bias_offsets[3],
                     d_fc2_out, nullptr);

        run_fc_layer(cublas, shape, 2, d_fc2_out,
                     d_weights + shape.weight_offsets[4],
                     d_biases  + shape.bias_offsets[4],
                     d_fc3_out, nullptr);
    };

    constexpr int NITERS = 100;
    float elapsed_ms = 0.0f;

    // Warmup
    if (opt.impl == "baseline") {
        run_baseline();
    } else {
        run_fused();
    }
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    // Timed loop
    check_cuda(cudaEventRecord(start), "record start");
    for (int iter = 0; iter < NITERS; ++iter) {
        if (opt.impl == "baseline") {
            run_baseline();
        } else {
            run_fused();
        }
    }
    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");
    elapsed_ms /= NITERS;

    check_cuda(cudaMemcpy(h_output.data(), d_fc3_out,
                          shape.output_elements * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy logits");

    if (!opt.dump_path.empty()) {
        std::ofstream ofs(opt.dump_path, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Failed to open dump path: " + opt.dump_path);
        }
        ofs.write(reinterpret_cast<const char*>(h_output.data()),
                  static_cast<std::streamsize>(h_output.size() * sizeof(float)));
        ofs.close();
    }

    bool verify_ok = true;
    if (opt.verify) {
        lenet_cpu_reference(opt,
                            shape,
                            h_weights,
                            shape.weight_offsets,
                            h_biases,
                            shape.bias_offsets,
                            h_input,
                            h_ref);

        float max_diff = 0.0f;
        for (size_t i = 0; i < h_output.size(); ++i) {
            float d = std::fabs(h_output[i] - h_ref[i]);
            if (d > max_diff) max_diff = d;
        }

        verify_ok = (max_diff <= 1e-3f);
        std::cout << "Verification max|diff| = " << std::scientific << max_diff
                  << (verify_ok ? " PASS" : " FAIL") << std::endl;
    }

    if (elapsed_ms > 0.0f) {
        std::cout << std::fixed << std::setprecision(2)
                  << "Impl=" << opt.impl
                  << " Batch=" << opt.batch
                  << " Algo=" << opt.algo
                  << " Time(ms)=" << elapsed_ms
                  << " GFLOP/s=" << lenet_gflops(shape, elapsed_ms)
                  << " Workspace(bytes)=" << ws_required
                  << std::endl;
    } else {
        std::cout << "Forward pass executed (timing TODO incomplete)." << std::endl;
    }

    // Cleanup
    check_cudnn(cudnnDestroyActivationDescriptor(identity_activation), "destroy identity activation");
    check_cudnn(cudnnDestroyTensorDescriptor(conv2_bias_desc), "destroy conv2_bias_desc");
    check_cudnn(cudnnDestroyTensorDescriptor(conv1_bias_desc), "destroy conv1_bias_desc");
    destroy_lenet_descriptors(descs);

    check_cuda(cudaEventDestroy(stop), "destroy stop");
    check_cuda(cudaEventDestroy(start), "destroy start");

    check_cublas(cublasDestroy(cublas), "cublasDestroy");
    check_cudnn(cudnnDestroy(cudnn), "cudnnDestroy");

    check_cuda(cudaFree(d_workspace), "free workspace");
    check_cuda(cudaFree(d_fc3_out), "free fc3_out");
    check_cuda(cudaFree(d_fc2_out), "free fc2_out");
    check_cuda(cudaFree(d_fc1_out), "free fc1_out");
    check_cuda(cudaFree(d_pool2_out), "free pool2_out");
    check_cuda(cudaFree(d_conv2_out), "free conv2_out");
    check_cuda(cudaFree(d_pool1_out), "free pool1_out");
    check_cuda(cudaFree(d_conv1_out), "free conv1_out");
    check_cuda(cudaFree(d_biases), "free biases");
    check_cuda(cudaFree(d_weights), "free weights");
    check_cuda(cudaFree(d_input), "free input");

    return verify_ok ? 0 : 2;
}