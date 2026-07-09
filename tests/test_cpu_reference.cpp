#include "cudabench/cpu_reference.h"

#include <cassert>
#include <iostream>

int main() {
    const cudabench::Matrix left{1.0F, 2.0F, 3.0F, 4.0F};
    const cudabench::Matrix right{5.0F, 6.0F, 7.0F, 8.0F};
    const auto product = cudabench::matrix_multiply_cpu(left, right, 2);
    assert(cudabench::nearly_equal(product, cudabench::Matrix{19.0F, 22.0F, 43.0F, 50.0F}));

    const cudabench::Matrix image{1.0F, 2.0F, 3.0F, 4.0F};
    const cudabench::Matrix identity{
        0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F};
    const auto filtered = cudabench::convolution2d_cpu(image, 2, 2, identity, 3);
    assert(cudabench::nearly_equal(filtered, image));

    assert(cudabench::nearly_equal(cudabench::reduce_sum_cpu(image), 10.0F));

    std::cout << "cpu_reference_tests=ok\n";
    return 0;
}
