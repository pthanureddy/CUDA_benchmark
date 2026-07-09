#include "cudabench/cuda_kernels.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cudabench {

namespace {

__global__ void matmul_kernel(const float* left, const float* right, float* output, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0F;
    for (int k = 0; k < n; ++k) {
        sum += left[row * n + k] * right[k * n + col];
    }
    output[row * n + col] = sum;
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

Matrix matrix_multiply_cuda(const Matrix& left, const Matrix& right, std::size_t n) {
    if (left.size() != n * n || right.size() != n * n) {
        throw std::invalid_argument("matrix_multiply_cuda expects two n x n matrices");
    }

    const std::size_t bytes = left.size() * sizeof(float);
    float* d_left = nullptr;
    float* d_right = nullptr;
    float* d_output = nullptr;
    check(cudaMalloc(&d_left, bytes), "cudaMalloc left");
    check(cudaMalloc(&d_right, bytes), "cudaMalloc right");
    check(cudaMalloc(&d_output, bytes), "cudaMalloc output");
    check(cudaMemcpy(d_left, left.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy left");
    check(cudaMemcpy(d_right, right.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy right");

    const dim3 block(16, 16);
    const dim3 grid((static_cast<unsigned int>(n) + block.x - 1) / block.x,
                    (static_cast<unsigned int>(n) + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(d_left, d_right, d_output, static_cast<int>(n));
    check(cudaGetLastError(), "matmul_kernel");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    Matrix output(left.size());
    check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy output");
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_output);
    return output;
}

}  // namespace cudabench
