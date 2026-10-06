# Heterogeneous CPU-GPU Matrix Multiplication Benchmark

Dense single-precision General Matrix Multiplication ($C = A \times B$, SGEMM) benchmarking suite spanning 8 distinct execution paradigms across host CPU and dedicated GPU architectures.

## Execution Paradigms

### CPU Implementations
1. **Sequential C++ (Baseline)**: Single-threaded CPU implementation.
2. **OpenMP Multithreaded C++**: Parallel loop distribution across CPU logical cores.
3. **Intel SYCL / oneAPI**: Data-parallel kernel with compiler-driven SIMD vectorization over Intel OpenCL runtime.
4. **CUDA on CPU (Emulation)**: Simulated thread block and grid hierarchy on host CPU.

### GPU Implementations (NVIDIA CUDA)
5. **Normal C++ on GPU (Naive 2D CUDA Kernel)**: 2D thread block decomposition directly on GPU cores.
6. **OpenMP on GPU (1D Grid-Stride CUDA Kernel)**: 1D thread distribution with persistent stride looping.
7. **SYCL on GPU (SYCL 2D Work-Item Style Kernel)**: 2D NDRange execution mapped to GPU streaming multiprocessors.
8. **CUDA Tiled on GPU (Shared-Memory Tiled Kernel)**: Optimized $16 \times 16$ tile caching in high-speed on-chip SRAM (`__shared__` memory).

---

## Project Structure

```
MatrixMultiplicationBenchmark/
├── BENCHMARK_REPORT.md         # Comprehensive performance analysis report
├── CMakeLists.txt              # CMake build configuration
├── README.md                   # Project overview and instructions
├── build_all.sh                # Compilation script (nvcc + icpx)
├── run_benchmark.sh            # Benchmark launcher script
├── unified_matrix_benchmark    # Pre-compiled 4-backend executable
└── unified_matrix_benchmark.cpp# Unified benchmark implementation
```

---

## Quick Start

### 1. Run the Benchmark
Execute the pre-built binary using the runner script:
```bash
./run_benchmark.sh [matrix_size] [omp_threads] [--only <sycl|cuda|openmp|cpu|gpu|all>]
```

**Examples:**
```bash
# Run default configuration (N=1024, all 8 implementations)
./run_benchmark.sh

# Run with 512x512 matrix size
./run_benchmark.sh 512

# Run GPU-only benchmarks with 1024x1024
./run_benchmark.sh 1024 20 --only gpu
```

### 2. Build from Source
Rebuild the unified executable using `nvcc` and `icpx`:
```bash
chmod +x build_all.sh run_benchmark.sh
./build_all.sh
```

---

## Hardware and Environment Requirements
- **Host CPU**: Multi-core x86_64 CPU (e.g. Intel Core i7-13650HX)
- **Dedicated GPU**: NVIDIA GPU with CUDA Compute Capability $\ge 5.0$ (e.g. GeForce RTX 4050)
- **Compilers**: 
  - NVIDIA CUDA Compiler (`nvcc` $\ge 12.0$)
  - Intel oneAPI DPC++/C++ Compiler (`icpx` $\ge 2024.0$)
- **Operating System**: Linux / WSL2 (Ubuntu 22.04+ recommended)