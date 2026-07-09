#include "cudabench/cpu_reference.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cudabench {

Matrix make_sequence(std::size_t count, float scale) {
    Matrix values(count);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * scale;
    }
    return values;
}

Matrix matrix_multiply_cpu(const Matrix& left, const Matrix& right, std::size_t n) {
    if (left.size() != n * n || right.size() != n * n) {
        throw std::invalid_argument("matrix_multiply_cpu expects two n x n matrices");
    }

    Matrix result(n * n, 0.0F);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            float sum = 0.0F;
            for (std::size_t k = 0; k < n; ++k) {
                sum += left[row * n + k] * right[k * n + col];
            }
            result[row * n + col] = sum;
        }
    }
    return result;
}

Matrix convolution2d_cpu(
    const Matrix& image,
    std::size_t width,
    std::size_t height,
    const Matrix& kernel,
    std::size_t kernel_size) {
    if (image.size() != width * height) {
        throw std::invalid_argument("convolution2d_cpu image size does not match dimensions");
    }
    if (kernel.size() != kernel_size * kernel_size || kernel_size % 2 == 0) {
        throw std::invalid_argument("convolution2d_cpu expects an odd square kernel");
    }

    Matrix output(width * height, 0.0F);
    const int radius = static_cast<int>(kernel_size / 2);
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            float sum = 0.0F;
            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    const int sx = std::clamp(static_cast<int>(x) + kx, 0, static_cast<int>(width) - 1);
                    const int sy = std::clamp(static_cast<int>(y) + ky, 0, static_cast<int>(height) - 1);
                    const auto kernel_index =
                        static_cast<std::size_t>(ky + radius) * kernel_size + static_cast<std::size_t>(kx + radius);
                    sum += image[static_cast<std::size_t>(sy) * width + static_cast<std::size_t>(sx)] *
                           kernel[kernel_index];
                }
            }
            output[y * width + x] = sum;
        }
    }
    return output;
}

float reduce_sum_cpu(const Matrix& values) {
    return std::accumulate(values.begin(), values.end(), 0.0F);
}

bool nearly_equal(const Matrix& left, const Matrix& right, float tolerance) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (!nearly_equal(left[index], right[index], tolerance)) {
            return false;
        }
    }
    return true;
}

bool nearly_equal(float left, float right, float tolerance) {
    return std::fabs(left - right) <= tolerance;
}

void write_results_csv(const std::string& path, const std::vector<BenchmarkResult>& results) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("unable to write benchmark results to " + path);
    }
    output << "workload,mode,problem_size,iterations,milliseconds,valid\n";
    output << std::fixed << std::setprecision(3);
    for (const auto& row : results) {
        output << row.workload << ',' << row.mode << ',' << row.problem_size << ',' << row.iterations << ','
               << row.milliseconds << ',' << (row.valid ? "true" : "false") << '\n';
    }
}

}  // namespace cudabench
