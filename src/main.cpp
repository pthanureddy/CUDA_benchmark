#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cudabench/cpu_reference.h"

#ifdef CUDABENCH_WITH_CUDA
#include "cudabench/cuda_kernels.h"
#endif

namespace {

void usage() {
    std::cerr
        << "usage:\n"
        << "  cuda_bench cpu-smoke\n"
        << "  cuda_bench matmul <n> <iterations>\n"
        << "  cuda_bench convolution <width> <height> <iterations>\n"
        << "  cuda_bench reduction <count> <iterations>\n";
}

std::size_t parse_size(const char* value) {
    return static_cast<std::size_t>(std::stoull(value));
}

int parse_iterations(const char* value) {
    const int parsed = std::stoi(value);
    if (parsed <= 0) {
        throw std::invalid_argument("iterations must be positive");
    }
    return parsed;
}

bool run_cpu_smoke() {
    const cudabench::Matrix left{1.0F, 2.0F, 3.0F, 4.0F};
    const cudabench::Matrix right{5.0F, 6.0F, 7.0F, 8.0F};
    const auto product = cudabench::matrix_multiply_cpu(left, right, 2);

    const cudabench::Matrix image{1.0F, 2.0F, 3.0F, 4.0F};
    const cudabench::Matrix identity_kernel{
        0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F};
    const auto filtered = cudabench::convolution2d_cpu(image, 2, 2, identity_kernel, 3);

    return cudabench::nearly_equal(product, cudabench::Matrix{19.0F, 22.0F, 43.0F, 50.0F}) &&
           cudabench::nearly_equal(filtered, image) &&
           cudabench::nearly_equal(cudabench::reduce_sum_cpu(image), 10.0F);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }

    try {
        const std::string command = argv[1];
        std::vector<cudabench::BenchmarkResult> results;

        if (command == "cpu-smoke") {
            if (!run_cpu_smoke()) {
                std::cerr << "cpu-smoke=failed\n";
                return 1;
            }
            std::cout << "cpu-smoke=ok\n";
            return 0;
        }

        if (command == "matmul") {
            if (argc != 4) {
                usage();
                return 2;
            }
            const auto n = parse_size(argv[2]);
            const int iterations = parse_iterations(argv[3]);
            const auto left = cudabench::make_sequence(n * n, 0.01F);
            const auto right = cudabench::make_sequence(n * n, 0.02F);
            cudabench::Matrix reference;
            results.push_back(cudabench::time_workload("matmul", "cpu", n, iterations, [&]() {
                reference = cudabench::matrix_multiply_cpu(left, right, n);
                return !reference.empty();
            }));
#ifdef CUDABENCH_WITH_CUDA
            results.push_back(cudabench::time_workload("matmul", "cuda", n, iterations, [&]() {
                const auto gpu = cudabench::matrix_multiply_cuda(left, right, n);
                return cudabench::nearly_equal(reference, gpu, 1.0e-2F);
            }));
#endif
        } else if (command == "convolution") {
            if (argc != 5) {
                usage();
                return 2;
            }
            const auto width = parse_size(argv[2]);
            const auto height = parse_size(argv[3]);
            const int iterations = parse_iterations(argv[4]);
            const auto image = cudabench::make_sequence(width * height, 0.01F);
            const cudabench::Matrix kernel{
                0.0F, -1.0F, 0.0F,
                -1.0F, 5.0F, -1.0F,
                0.0F, -1.0F, 0.0F};
            cudabench::Matrix reference;
            results.push_back(cudabench::time_workload("convolution", "cpu", width * height, iterations, [&]() {
                reference = cudabench::convolution2d_cpu(image, width, height, kernel, 3);
                return !reference.empty();
            }));
#ifdef CUDABENCH_WITH_CUDA
            results.push_back(cudabench::time_workload("convolution", "cuda", width * height, iterations, [&]() {
                const auto gpu = cudabench::convolution2d_cuda(image, width, height, kernel, 3);
                return cudabench::nearly_equal(reference, gpu, 1.0e-2F);
            }));
#endif
        } else if (command == "reduction") {
            if (argc != 4) {
                usage();
                return 2;
            }
            const auto count = parse_size(argv[2]);
            const int iterations = parse_iterations(argv[3]);
            const auto values = cudabench::make_sequence(count, 0.01F);
            float reference = 0.0F;
            results.push_back(cudabench::time_workload("reduction", "cpu", count, iterations, [&]() {
                reference = cudabench::reduce_sum_cpu(values);
                return true;
            }));
#ifdef CUDABENCH_WITH_CUDA
            results.push_back(cudabench::time_workload("reduction", "cuda", count, iterations, [&]() {
                const auto gpu = cudabench::reduce_sum_cuda(values);
                return cudabench::nearly_equal(reference, gpu, 1.0e-2F);
            }));
#endif
        } else {
            usage();
            return 2;
        }

        cudabench::write_results_csv("benchmark_results.csv", results);
        for (const auto& result : results) {
            std::cout << result.workload << "," << result.mode << "," << result.problem_size << ","
                      << result.iterations << "," << result.milliseconds << ","
                      << (result.valid ? "valid" : "invalid") << "\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
