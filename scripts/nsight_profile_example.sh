#!/usr/bin/env bash
set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
  echo "Nsight Systems CLI (nsys) is not available on PATH." >&2
  exit 1
fi

nsys profile --stats=true --output cuda_bench_profile ./build/cuda_bench matmul 1024 10
