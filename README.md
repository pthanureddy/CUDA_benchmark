# CUDA Parallel Computing Benchmarking Suite

A portfolio-ready CUDA/C++ benchmarking project built around three classic GPU workloads:

- matrix multiplication (**naive vs shared-memory tiled**)
- image convolution filters
- parallel reduction

It includes CPU reference implementations, optional CUDA builds, Linux-friendly CMake setup, correctness validation, and simple result plotting.

## Why this repo works for applications

- Demonstrates real GPU programming concepts instead of only theory
- Shows side-by-side **CPU baseline vs GPU kernels**
- Includes **floating-point correctness validation** against CPU reference outputs
- Keeps the project usable even on machines **without a local GPU**
- Adds a practical path for **Nsight Systems profiling** and benchmark export

## Project structure

- `src/cpu_reference.cpp` — CPU baseline implementations
- `src/matrix_mul.cu` — naive and shared-memory matrix multiplication kernels
- `src/convolution.cu` — 2D convolution kernel
- `src/reduction.cu` — tree-style parallel reduction kernel
- `src/main.cpp` — CLI entry point
- `python/plot_results.py` — turn benchmark CSV output into a chart
- `scripts/nsight_profile_example.sh` — example profiling command

## Build CPU-only mode

This works on any Linux machine with a C++ compiler and CMake:

```bash
cmake -S . -B build -DCUDABENCH_ENABLE_CUDA=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Build with CUDA

On a machine with `nvcc` available:

```bash
cmake -S . -B build -DCUDABENCH_ENABLE_CUDA=ON
cmake --build build --parallel
```

## Run examples

CPU smoke test:

```bash
./build/cuda_bench cpu-smoke
```

Matrix multiplication benchmark:

```bash
./build/cuda_bench matmul 1024 20
```

Convolution benchmark:

```bash
./build/cuda_bench convolution 1024 1024 20
```

Reduction benchmark:

```bash
./build/cuda_bench reduction 1048576 50
```

Results are written to `benchmark_results.csv`, which you can plot with:

```bash
python3 python/plot_results.py benchmark_results.csv
```

## Suggested GitHub repo description

> CUDA benchmarking suite with matrix multiplication, convolution, and reduction kernels; CPU baselines, validation, plotting, and Linux CMake setup.

## Suggested CV line

**CUDA Parallel Computing Benchmarking Suite** | CUDA C++, C++, Python, Linux | 2026  
Implemented CUDA kernels for matrix multiplication, convolution, and reduction; benchmarked CPU vs GPU performance, validated outputs against CPU references, and added scripts for profiling and result visualization.

## Honest scope note

GitHub-hosted CI in this repo runs CPU-only smoke tests. Full GPU benchmarking requires a CUDA-capable machine, Google Colab with GPU enabled, or a self-hosted runner with NVIDIA tooling installed.
