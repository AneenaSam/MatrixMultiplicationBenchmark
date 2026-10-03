#!/usr/bin/env bash
# run_benchmark.sh — Unified Matrix Multiplication Benchmark launcher
# Supports all 4 backends: Normal C++, OpenMP CPU, Intel SYCL, NVIDIA CUDA GPU

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================================================
# Detect Intel oneAPI installation
# =========================================================
ONEAPI_ROOT="/opt/intel/oneapi"
ONEAPI_SETVARS="${ONEAPI_ROOT}/setvars.sh"

if [[ -f "$ONEAPI_SETVARS" ]]; then
    source "$ONEAPI_SETVARS" > /dev/null 2>&1 || true
fi

# Locate required library directories
UMF_LIB=$(find "${ONEAPI_ROOT}" -name 'libumf.so.1' 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
REDIST_LIB="${ONEAPI_ROOT}/redist/lib"
COMPILER_LIB=$(find "${ONEAPI_ROOT}/compiler" -name 'libiomp5.so' 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
CUDA_LIB="/usr/local/cuda/lib64"

# Build LD_LIBRARY_PATH
SYCL_LIBS=""
[[ -n "$UMF_LIB" ]]        && SYCL_LIBS="${UMF_LIB}:${SYCL_LIBS}"
[[ -d "$REDIST_LIB" ]]     && SYCL_LIBS="${REDIST_LIB}:${SYCL_LIBS}"
[[ -n "$COMPILER_LIB" ]]   && SYCL_LIBS="${COMPILER_LIB}:${SYCL_LIBS}"
[[ -d "$CUDA_LIB" ]]       && SYCL_LIBS="${CUDA_LIB}:${SYCL_LIBS}"
[[ -n "$LD_LIBRARY_PATH" ]]&& SYCL_LIBS="${SYCL_LIBS}${LD_LIBRARY_PATH}"

# =========================================================
# Detect which binary to use (prioritizing 4-backend binary)
# =========================================================
FULL_BIN="${SCRIPT_DIR}/unified_matrix_benchmark"
SYCL_BIN="${SCRIPT_DIR}/unified_sycl"
STD_BIN="${SCRIPT_DIR}/build/unified_matrix_benchmark"

if [[ -f "$FULL_BIN" ]]; then
    exec env LD_LIBRARY_PATH="${SYCL_LIBS}" "$FULL_BIN" "$@"
elif [[ -f "$SYCL_BIN" ]]; then
    exec env LD_LIBRARY_PATH="${SYCL_LIBS}" "$SYCL_BIN" "$@"
elif [[ -f "$STD_BIN" ]]; then
    exec "$STD_BIN" "$@"
else
    echo "[ERROR] No benchmark binary found. Please build the project first:"
    echo "  ./build_all.sh"
    exit 1
fi
