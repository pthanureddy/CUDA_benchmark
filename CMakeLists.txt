cmake_minimum_required(VERSION 3.20)
project(cuda_benchmark_suite VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CUDABENCH_ENABLE_CUDA "Build CUDA kernels and GPU benchmark executable" ON)

add_library(cudabench_cpu
    src/cpu_reference.cpp
)
target_include_directories(cudabench_cpu PUBLIC include)

if(MSVC)
  target_compile_options(cudabench_cpu PRIVATE /W4 /permissive-)
else()
  target_compile_options(cudabench_cpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(cpu_reference_tests tests/test_cpu_reference.cpp)
target_link_libraries(cpu_reference_tests PRIVATE cudabench_cpu)

add_executable(cuda_bench src/main.cpp)
target_link_libraries(cuda_bench PRIVATE cudabench_cpu)
target_include_directories(cuda_bench PRIVATE include)

enable_testing()
add_test(NAME cpu_reference_tests COMMAND cpu_reference_tests)
add_test(NAME cpu_smoke COMMAND cuda_bench cpu-smoke)

if(CUDABENCH_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_library(cudabench_gpu
        src/matrix_mul.cu
        src/convolution.cu
        src/reduction.cu
    )
    target_include_directories(cudabench_gpu PUBLIC include)
    target_compile_definitions(cudabench_gpu PUBLIC CUDABENCH_WITH_CUDA)
    target_compile_definitions(cuda_bench PRIVATE CUDABENCH_WITH_CUDA)
    set_target_properties(cudabench_gpu PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    target_link_libraries(cuda_bench PRIVATE cudabench_gpu cudabench_cpu)
  else()
    message(STATUS "CUDA compiler not found; building CPU-only targets.")
  endif()
endif()
