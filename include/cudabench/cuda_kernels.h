#pragma once

#include <cstddef>

#include "cudabench/cpu_reference.h"

namespace cudabench {

[[nodiscard]] Matrix matrix_multiply_cuda(const Matrix& left, const Matrix& right, std::size_t n);
[[nodiscard]] Matrix convolution2d_cuda(
    const Matrix& image,
    std::size_t width,
    std::size_t height,
    const Matrix& kernel,
    std::size_t kernel_size);
[[nodiscard]] float reduce_sum_cuda(const Matrix& values);

}  // namespace cudabench
