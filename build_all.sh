#!/usr/bin/env bash
# build_all.sh — Compiles all 4 backends (Normal C++, OpenMP, SYCL, CUDA) into unified_matrix_benchmark
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==========================================================="
echo " Building Unified Matrix Multiplication Benchmark (4 Backends)"
echo "==========================================================="

# Source Intel oneAPI environment
if [[ -f "/opt/intel/oneapi/setvars.sh" ]]; then
    source "/opt/intel/oneapi/setvars.sh" > /dev/null 2>&1 || true
fi

# Detect CUDA nvcc
NVCC="/usr/local/cuda-13.3/bin/nvcc"
if [[ ! -x "$NVCC" ]]; then
    NVCC=$(which nvcc 2>/dev/null || echo "")
fi

if [[ -z "$NVCC" ]]; then
    echo "[ERROR] nvcc compiler not found!"
    exit 1
fi

echo "[1/2] Compiling CUDA component with nvcc ($NVCC)..."
"$NVCC" -O3 -arch=sm_89 -diag-suppress 177 -x cu -c unified_matrix_benchmark.cpp -DUSE_CUDA -DMATMUL_NO_MAIN -o cuda_part.o

echo "[2/2] Compiling C++, OpenMP, SYCL and linking with icpx..."
icpx -O3 -std=c++17 -fsycl -DUSE_SYCL -DUSE_OPENMP -DUSE_CUDA -qopenmp \
    unified_matrix_benchmark.cpp cuda_part.o \
    -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart \
    -o unified_matrix_benchmark

echo "==========================================================="
echo " Build successful! Executable: ./unified_matrix_benchmark"
echo " Run with: ./run_benchmark.sh [matrix_size] [omp_threads]"
echo "==========================================================="
