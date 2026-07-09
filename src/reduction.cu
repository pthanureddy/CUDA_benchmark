#include "cudabench/cuda_kernels.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cudabench {

namespace {

__global__ void reduction_kernel(const float* values, float* partials, int count) {
    extern __shared__ float shared[];
    const int tid = threadIdx.x;
    const int global = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0F;
    if (global < count) {
        sum += values[global];
    }
    if (global + blockDim.x < count) {
        sum += values[global + blockDim.x];
    }
    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = shared[0];
    }
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

float reduce_sum_cuda(const Matrix& values) {
    if (values.empty()) {
        return 0.0F;
    }

    const int block_size = 256;
    const int blocks = static_cast<int>((values.size() + block_size * 2 - 1) / (block_size * 2));
    const std::size_t input_bytes = values.size() * sizeof(float);
    const std::size_t partial_bytes = static_cast<std::size_t>(blocks) * sizeof(float);

    float* d_values = nullptr;
    float* d_partials = nullptr;
    check(cudaMalloc(&d_values, input_bytes), "cudaMalloc values");
    check(cudaMalloc(&d_partials, partial_bytes), "cudaMalloc partials");
    check(cudaMemcpy(d_values, values.data(), input_bytes, cudaMemcpyHostToDevice), "cudaMemcpy values");

    reduction_kernel<<<blocks, block_size, block_size * sizeof(float)>>>(
        d_values,
        d_partials,
        static_cast<int>(values.size()));
    check(cudaGetLastError(), "reduction_kernel");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    Matrix partials(static_cast<std::size_t>(blocks));
    check(cudaMemcpy(partials.data(), d_partials, partial_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy partials");
    cudaFree(d_values);
    cudaFree(d_partials);

    float total = 0.0F;
    for (float partial : partials) {
        total += partial;
    }
    return total;
}

}  // namespace cudabench
