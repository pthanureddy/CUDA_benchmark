#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace cudabench {

using Matrix = std::vector<float>;

struct BenchmarkResult {
    std::string workload;
    std::string mode;
    std::size_t problem_size{0};
    int iterations{0};
    double milliseconds{0.0};
    bool valid{false};
};

[[nodiscard]] Matrix make_sequence(std::size_t count, float scale = 1.0F);
[[nodiscard]] Matrix matrix_multiply_cpu(const Matrix& left, const Matrix& right, std::size_t n);
[[nodiscard]] Matrix convolution2d_cpu(
    const Matrix& image,
    std::size_t width,
    std::size_t height,
    const Matrix& kernel,
    std::size_t kernel_size);
[[nodiscard]] float reduce_sum_cpu(const Matrix& values);
[[nodiscard]] bool nearly_equal(const Matrix& left, const Matrix& right, float tolerance = 1.0e-3F);
[[nodiscard]] bool nearly_equal(float left, float right, float tolerance = 1.0e-3F);

void write_results_csv(const std::string& path, const std::vector<BenchmarkResult>& results);

template <typename Fn>
BenchmarkResult time_workload(
    std::string workload,
    std::string mode,
    std::size_t problem_size,
    int iterations,
    Fn&& fn) {
    const auto start = std::chrono::steady_clock::now();
    bool valid = true;
    for (int index = 0; index < iterations; ++index) {
        valid = fn() && valid;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    return BenchmarkResult{std::move(workload), std::move(mode), problem_size, iterations, elapsed, valid};
}

}  // namespace cudabench
