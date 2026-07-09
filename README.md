# CUDA Parallel Computing Benchmarking Suite

CUDA/C++ benchmarking project for three GPU-oriented workloads:

- matrix multiplication
- image convolution filters
- parallel reduction

The repository includes CPU reference implementations, optional CUDA kernels,
correctness validation against CPU outputs, CSV benchmark export, a plotting
script, and Linux CI that verifies the CPU-only build path.

## Why This Project Exists

The project demonstrates practical GPU programming concepts while staying
usable on machines without a local NVIDIA GPU. CI runs the CPU reference build
and smoke tests; full CUDA timing requires a CUDA-capable machine, Google Colab
with GPU enabled, or a self-hosted runner with NVIDIA tooling.

## Build CPU-Only Mode

```bash
cmake -S . -B build -DCUDABENCH_ENABLE_CUDA=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Build With CUDA

```bash
cmake -S . -B build -DCUDABENCH_ENABLE_CUDA=ON
cmake --build build --parallel
```

## Run Examples

```bash
./build/cuda_bench cpu-smoke
./build/cuda_bench matmul 1024 20
./build/cuda_bench convolution 1024 1024 20
./build/cuda_bench reduction 1048576 50
```

Results are written to `benchmark_results.csv`.

```bash
python3 python/plot_results.py benchmark_results.csv
```

## Repository Layout

```text
include/cudabench/     CPU and CUDA benchmark APIs
src/cpu_reference.cpp  CPU baselines and CSV export
src/*.cu               Optional CUDA kernels
src/main.cpp           CLI benchmark runner
tests/                 CPU correctness smoke tests
python/                Result plotting script
scripts/               Nsight profiling helper
.github/workflows/     Ubuntu CPU-only CMake build and test workflow
```

## Honest Scope Note

GitHub-hosted CI in this repository runs CPU-only correctness and build tests.
The CUDA kernels are included and build when `nvcc` is available, but this repo
does not publish GPU speedup numbers unless they are measured on a CUDA-capable
machine.

## Resume-Ready Summary

Built a CUDA/C++ benchmark suite with CPU references, optional CUDA kernels for
matrix multiplication, image convolution, and parallel reduction, correctness
validation, CSV result export, plotting support, and Linux CPU-only GitHub
Actions CI.
