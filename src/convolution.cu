#include "cudabench/cuda_kernels.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cudabench {

namespace {

__global__ void convolution_kernel(
    const float* image,
    int width,
    int height,
    const float* kernel,
    int kernel_size,
    float* output) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int radius = kernel_size / 2;
    float sum = 0.0F;
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            const int unclamped_x = x + kx;
            const int unclamped_y = y + ky;
            const int sx = unclamped_x < 0 ? 0 : (unclamped_x >= width ? width - 1 : unclamped_x);
            const int sy = unclamped_y < 0 ? 0 : (unclamped_y >= height ? height - 1 : unclamped_y);
            const int kernel_index = (ky + radius) * kernel_size + (kx + radius);
            sum += image[sy * width + sx] * kernel[kernel_index];
        }
    }
    output[y * width + x] = sum;
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

Matrix convolution2d_cuda(
    const Matrix& image,
    std::size_t width,
    std::size_t height,
    const Matrix& kernel,
    std::size_t kernel_size) {
    if (image.size() != width * height || kernel.size() != kernel_size * kernel_size) {
        throw std::invalid_argument("convolution2d_cuda input sizes do not match dimensions");
    }

    const std::size_t image_bytes = image.size() * sizeof(float);
    const std::size_t kernel_bytes = kernel.size() * sizeof(float);
    float* d_image = nullptr;
    float* d_kernel = nullptr;
    float* d_output = nullptr;
    check(cudaMalloc(&d_image, image_bytes), "cudaMalloc image");
    check(cudaMalloc(&d_kernel, kernel_bytes), "cudaMalloc kernel");
    check(cudaMalloc(&d_output, image_bytes), "cudaMalloc output");
    check(cudaMemcpy(d_image, image.data(), image_bytes, cudaMemcpyHostToDevice), "cudaMemcpy image");
    check(cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice), "cudaMemcpy kernel");

    const dim3 block(16, 16);
    const dim3 grid((static_cast<unsigned int>(width) + block.x - 1) / block.x,
                    (static_cast<unsigned int>(height) + block.y - 1) / block.y);
    convolution_kernel<<<grid, block>>>(
        d_image,
        static_cast<int>(width),
        static_cast<int>(height),
        d_kernel,
        static_cast<int>(kernel_size),
        d_output);
    check(cudaGetLastError(), "convolution_kernel");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    Matrix output(image.size());
    check(cudaMemcpy(output.data(), d_output, image_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy output");
    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
    return output;
}

}  // namespace cudabench
