/*
 * unified_matrix_benchmark.cpp
 * Unified CPU-GPU Matrix Multiplication Benchmark for Ubuntu Linux
 * Supports: Normal C++, OpenMP C++, Intel SYCL GPU, NVIDIA CUDA GPU
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <thread>
#include <cstring>
#include <numeric>
#include <set>

#include <unistd.h>
#include <sys/sysinfo.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct MethodResult {
    std::string name;
    bool available = false;
    std::string device_info;
    int threads_used = 1;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double gflops = 0.0;
    double speedup = 0.0;
    bool verified = false;
    float max_abs_error = 0.0f;
    
    // Transfer timing for GPUs
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double e2e_ms = 0.0;
    bool has_transfer_times = false;
};

// ============================================================================
// HARDWARE DETECTION HELPERS (Linux / Ubuntu)
// ============================================================================

static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

static std::string get_cpu_model() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    if (cpuinfo.is_open()) {
        while (std::getline(cpuinfo, line)) {
            if (line.rfind("model name", 0) == 0) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    return trim(line.substr(colon + 1));
                }
            }
        }
    }
    return "Unknown CPU";
}

static std::string get_cpu_vendor() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    if (cpuinfo.is_open()) {
        while (std::getline(cpuinfo, line)) {
            if (line.rfind("vendor_id", 0) == 0) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    return trim(line.substr(colon + 1));
                }
            }
        }
    }
    return "Unknown Vendor";
}

static void get_cpu_counts(int& physical_cores, int& logical_cpus) {
    logical_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (logical_cpus <= 0) {
        logical_cpus = std::thread::hardware_concurrency();
    }
    
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    std::set<std::pair<int, int>> core_set; // (physical id, core id)
    int current_phys_id = -1;
    int current_core_id = -1;
    
    if (cpuinfo.is_open()) {
        while (std::getline(cpuinfo, line)) {
            if (line.rfind("physical id", 0) == 0) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    current_phys_id = std::atoi(line.substr(colon + 1).c_str());
                }
            } else if (line.rfind("core id", 0) == 0) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    current_core_id = std::atoi(line.substr(colon + 1).c_str());
                }
            }
            if (current_phys_id != -1 && current_core_id != -1) {
                core_set.insert({current_phys_id, current_core_id});
                current_phys_id = -1;
                current_core_id = -1;
            }
        }
    }
    
    if (!core_set.empty()) {
        physical_cores = static_cast<int>(core_set.size());
    } else {
        physical_cores = logical_cpus / 2 > 0 ? logical_cpus / 2 : 1;
    }
}

static std::string read_sysfs_file(const std::string& path) {
    std::ifstream file(path);
    std::string val;
    if (file.is_open()) {
        std::getline(file, val);
        return trim(val);
    }
    return "";
}

static void get_cpu_frequencies(std::string& base_freq, std::string& max_freq) {
    std::string min_val = read_sysfs_file("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq");
    std::string max_val = read_sysfs_file("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    
    if (!min_val.empty()) {
        double mhz = std::atof(min_val.c_str()) / 1000.0;
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << (mhz >= 1000 ? mhz/1000.0 : mhz) << (mhz >= 1000 ? " GHz" : " MHz");
        base_freq = ss.str();
    } else {
        base_freq = "N/A";
    }
    
    if (!max_val.empty()) {
        double mhz = std::atof(max_val.c_str()) / 1000.0;
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << (mhz >= 1000 ? mhz/1000.0 : mhz) << (mhz >= 1000 ? " GHz" : " MHz");
        max_freq = ss.str();
    } else {
        max_freq = "N/A";
    }
}

static void get_cache_info(std::string& l1d, std::string& l1i, std::string& l2, std::string& l3) {
    l1d = l1i = l2 = l3 = "N/A";
    for (int i = 0; i < 5; ++i) {
        std::string base = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(i) + "/";
        std::string level = read_sysfs_file(base + "level");
        std::string type = read_sysfs_file(base + "type");
        std::string size = read_sysfs_file(base + "size");
        
        if (level == "1") {
            if (type == "Data") l1d = size;
            else if (type == "Instruction") l1i = size;
        } else if (level == "2") {
            l2 = size;
        } else if (level == "3") {
            l3 = size;
        }
    }
}

static void get_ram_info(std::string& total_ram, std::string& avail_ram) {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        double total_gb = static_cast<double>(info.totalram * info.mem_unit) / (1024.0 * 1024.0 * 1024.0);
        double avail_gb = static_cast<double>(info.freeram * info.mem_unit) / (1024.0 * 1024.0 * 1024.0);
        
        std::ostringstream ss1, ss2;
        ss1 << std::fixed << std::setprecision(2) << total_gb << " GB";
        ss2 << std::fixed << std::setprecision(2) << avail_gb << " GB";
        total_ram = ss1.str();
        avail_ram = ss2.str();
    } else {
        total_ram = "N/A";
        avail_ram = "N/A";
    }
}

// ============================================================================
// MATRIX MULTIPLICATION CORE & VERIFICATION
// ============================================================================

static void init_matrices(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, int N) {
    const size_t total = static_cast<size_t>(N) * N;
    A.resize(total);
    B.resize(total);
    C.assign(total, 0.0f);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = static_cast<float>((i % 4) + 1);
            B[i * N + j] = static_cast<float>((j % 3) + 1);
        }
    }
}

static void calculate_statistics(const std::vector<double>& times_ms, MethodResult& res, int N, double ref_mean_ms) {
    if (times_ms.empty()) return;
    
    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());
    
    res.min_ms = sorted.front();
    res.max_ms = sorted.back();
    
    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    res.mean_ms = sum / sorted.size();
    
    size_t sz = sorted.size();
    if (sz % 2 == 0) {
        res.median_ms = (sorted[sz / 2 - 1] + sorted[sz / 2]) / 2.0;
    } else {
        res.median_ms = sorted[sz / 2];
    }
    
    // 2 * N^3 floating-point operations
    double ops = 2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    res.gflops = (ops / (res.mean_ms / 1000.0)) / 1e9;
    
    if (ref_mean_ms > 0.0 && res.verified) {
        res.speedup = ref_mean_ms / res.mean_ms;
    } else {
        res.speedup = (res.name == "Normal C++") ? 1.0 : 0.0;
    }
}

static void verify_matrices(const std::vector<float>& ref, const std::vector<float>& test, int N, MethodResult& res, float tol = 1e-3f) {
    res.verified = true;
    res.max_abs_error = 0.0f;
    const size_t total = static_cast<size_t>(N) * N;
    
    for (size_t i = 0; i < total; ++i) {
        float err = std::abs(ref[i] - test[i]);
        if (err > res.max_abs_error) {
            res.max_abs_error = err;
        }
        if (err > tol) {
            res.verified = false;
        }
    }
}

// ============================================================================
// METHOD 1: NORMAL C++ (Sequential)
// ============================================================================

static void matmul_normal_cpp_kernel(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static MethodResult run_normal_cpp(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C_ref, int N, int warmups, int runs) {
    MethodResult res;
    res.name = "Normal C++";
    res.available = true;
    res.device_info = get_cpu_model() + " (1 thread)";
    res.threads_used = 1;
    
    // Warm-up runs
    for (int w = 0; w < warmups; ++w) {
        std::fill(C_ref.begin(), C_ref.end(), 0.0f);
        matmul_normal_cpp_kernel(A.data(), B.data(), C_ref.data(), N);
    }
    
    // Measured runs
    std::vector<double> times_ms;
    times_ms.reserve(runs);
    
    for (int r = 0; r < runs; ++r) {
        std::fill(C_ref.begin(), C_ref.end(), 0.0f);
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_normal_cpp_kernel(A.data(), B.data(), C_ref.data(), N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(elapsed_ms);
    }
    
    res.verified = true;
    res.max_abs_error = 0.0f;
    calculate_statistics(times_ms, res, N, 0.0);
    return res;
}

// ============================================================================
// METHOD 2: OPENMP CPU
// ============================================================================

#ifdef USE_OPENMP
static void matmul_openmp_kernel(const float* A, const float* B, float* C, int N, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
#endif

static MethodResult run_openmp(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C_ref, int N, int threads_req, int warmups, int runs, double ref_mean_ms) {
    MethodResult res;
    res.name = "OpenMP";

#ifdef USE_OPENMP
    int max_threads = omp_get_max_threads();
    int threads_used = (threads_req > 0) ? std::min(threads_req, max_threads) : max_threads;
    
    res.available = true;
    res.threads_used = threads_used;
    res.device_info = get_cpu_model() + " (" + std::to_string(threads_used) + " threads)";
    
    std::vector<float> C(N * N, 0.0f);
    
    // Warm-up runs
    for (int w = 0; w < warmups; ++w) {
        std::fill(C.begin(), C.end(), 0.0f);
        matmul_openmp_kernel(A.data(), B.data(), C.data(), N, threads_used);
    }
    
    // Measured runs
    std::vector<double> times_ms;
    times_ms.reserve(runs);
    
    for (int r = 0; r < runs; ++r) {
        std::fill(C.begin(), C.end(), 0.0f);
        double t0 = omp_get_wtime();
        matmul_openmp_kernel(A.data(), B.data(), C.data(), N, threads_used);
        double t1 = omp_get_wtime();
        times_ms.push_back((t1 - t0) * 1000.0);
    }
    
    verify_matrices(C_ref, C, N, res);
    calculate_statistics(times_ms, res, N, ref_mean_ms);
#else
    res.available = false;
    res.device_info = "OpenMP compiler support not enabled (-DUSE_OPENMP -fopenmp)";
#endif

    return res;
}

// ============================================================================
// METHOD 3: INTEL SYCL
// ============================================================================

#ifdef USE_SYCL
static MethodResult run_sycl(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C_ref, int N, int warmups, int runs, double ref_mean_ms) {
    MethodResult res;
    res.name = "SYCL";

    try {
        // Smart device selection: try GPU first, fall back to any available device (CPU)
        sycl::device chosen_dev;
        bool gpu_found = false;

        // Try to find any GPU device from any platform
        for (auto& plat : sycl::platform::get_platforms()) {
            for (auto& dev : plat.get_devices()) {
                if (dev.is_gpu()) {
                    chosen_dev = dev;
                    gpu_found = true;
                    break;
                }
            }
            if (gpu_found) break;
        }

        // If no GPU found, fall back to CPU
        if (!gpu_found) {
            for (auto& plat : sycl::platform::get_platforms()) {
                for (auto& dev : plat.get_devices()) {
                    if (dev.is_cpu()) {
                        chosen_dev = dev;
                        break;
                    }
                }
            }
        }

        sycl::queue q{chosen_dev, sycl::property::queue::enable_profiling{}};
        sycl::device dev = q.get_device();
        sycl::platform plat = dev.get_platform();

        res.available = true;
        std::string dev_name = dev.get_info<sycl::info::device::name>();
        std::string plat_name = plat.get_info<sycl::info::platform::name>();
        std::string dev_type = dev.is_gpu() ? " [GPU]" : " [CPU]";
        res.device_info = dev_name + dev_type + " via " + plat_name;

        const size_t total_elements = static_cast<size_t>(N) * N;
        const size_t bytes = total_elements * sizeof(float);

        // USM Device Allocations
        float* d_A = sycl::malloc_device<float>(total_elements, q);
        float* d_B = sycl::malloc_device<float>(total_elements, q);
        float* d_C = sycl::malloc_device<float>(total_elements, q);

        if (!d_A || !d_B || !d_C) {
            if (d_A) sycl::free(d_A, q);
            if (d_B) sycl::free(d_B, q);
            if (d_C) sycl::free(d_C, q);
            res.available = false;
            res.device_info = "SYCL USM Memory allocation failed";
            return res;
        }

        // Measure Host-to-Device transfer
        auto t_h2d_0 = std::chrono::high_resolution_clock::now();
        q.memcpy(d_A, A.data(), bytes).wait();
        q.memcpy(d_B, B.data(), bytes).wait();
        auto t_h2d_1 = std::chrono::high_resolution_clock::now();
        res.h2d_ms = std::chrono::duration<double, std::milli>(t_h2d_1 - t_h2d_0).count();

        // Kernel Warm-up
        for (int w = 0; w < warmups; ++w) {
            q.memset(d_C, 0, bytes).wait();
            q.parallel_for(sycl::range<2>(N, N), [=](sycl::id<2> item) {
                int row = item[0];
                int col = item[1];
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += d_A[row * N + k] * d_B[k * N + col];
                }
                d_C[row * N + col] = sum;
            }).wait();
        }

        // Measured Kernel Runs
        std::vector<double> times_ms;
        times_ms.reserve(runs);

        for (int r = 0; r < runs; ++r) {
            q.memset(d_C, 0, bytes).wait();
            sycl::event e = q.parallel_for(sycl::range<2>(N, N), [=](sycl::id<2> item) {
                int row = item[0];
                int col = item[1];
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += d_A[row * N + k] * d_B[k * N + col];
                }
                d_C[row * N + col] = sum;
            });
            e.wait();
            auto start_ns = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto end_ns = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            double elapsed_ms = static_cast<double>(end_ns - start_ns) / 1e6;
            times_ms.push_back(elapsed_ms);
        }

        std::vector<float> C(total_elements, 0.0f);
        auto t_d2h_0 = std::chrono::high_resolution_clock::now();
        q.memcpy(C.data(), d_C, bytes).wait();
        auto t_d2h_1 = std::chrono::high_resolution_clock::now();
        res.d2h_ms = std::chrono::duration<double, std::milli>(t_d2h_1 - t_d2h_0).count();

        // Cleanup SYCL Memory
        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);

        verify_matrices(C_ref, C, N, res);
        calculate_statistics(times_ms, res, N, ref_mean_ms);

        res.kernel_ms = res.mean_ms;
        res.e2e_ms = res.h2d_ms + res.kernel_ms + res.d2h_ms;
        res.has_transfer_times = true;

    } catch (const sycl::exception& ex) {
        res.available = false;
        res.device_info = std::string("SYCL exception: ") + ex.what();
    } catch (const std::exception& ex) {
        res.available = false;
        res.device_info = std::string("SYCL error: ") + ex.what();
    }

    return res;
}
#else
static MethodResult run_sycl(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, int, int, int, double) {
    MethodResult res;
    res.name = "SYCL";
    res.available = false;
    res.device_info = "SYCL compiler support not enabled (-DUSE_SYCL -fsycl)";
    return res;
}
#endif

// ============================================================================
// METHOD 4: NVIDIA CUDA
// ============================================================================

#if defined(USE_CUDA) && defined(__CUDACC__)
__global__ void matmul_cuda_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

MethodResult run_cuda(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C_ref, int N, int warmups, int runs, double ref_mean_ms) {
    MethodResult res;
    res.name = "CUDA";

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        res.available = false;
        res.device_info = "No CUDA supported GPU found";
        return res;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    res.available = true;
    res.device_info = std::string(prop.name) + " (Compute " + std::to_string(prop.major) + "." + std::to_string(prop.minor) + ")";

    const size_t total_elements = static_cast<size_t>(N) * N;
    const size_t bytes = total_elements * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (cudaMalloc(&d_A, bytes) != cudaSuccess ||
        cudaMalloc(&d_B, bytes) != cudaSuccess ||
        cudaMalloc(&d_C, bytes) != cudaSuccess) {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        res.available = false;
        res.device_info = "CUDA Memory allocation failed";
        return res;
    }

    // Measure Host-to-Device transfer
    cudaEvent_t h2d_start, h2d_stop;
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_stop);
    
    cudaEventRecord(h2d_start);
    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(h2d_stop);
    cudaEventSynchronize(h2d_stop);
    
    float h2d_elapsed = 0.0f;
    cudaEventElapsedTime(&h2d_elapsed, h2d_start, h2d_stop);
    res.h2d_ms = h2d_elapsed;
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_stop);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Warm-up runs
    for (int w = 0; w < warmups; ++w) {
        cudaMemset(d_C, 0, bytes);
        matmul_cuda_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
    }

    // Measured runs with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<double> times_ms;
    times_ms.reserve(runs);

    for (int r = 0; r < runs; ++r) {
        cudaMemset(d_C, 0, bytes);
        cudaEventRecord(start);
        matmul_cuda_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        times_ms.push_back(milliseconds);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Device-to-Host transfer
    std::vector<float> C(total_elements, 0.0f);
    cudaEvent_t d2h_start, d2h_stop;
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_stop);
    
    cudaEventRecord(d2h_start);
    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(d2h_stop);
    cudaEventSynchronize(d2h_stop);
    
    float d2h_elapsed = 0.0f;
    cudaEventElapsedTime(&d2h_elapsed, d2h_start, d2h_stop);
    res.d2h_ms = d2h_elapsed;
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    verify_matrices(C_ref, C, N, res);
    calculate_statistics(times_ms, res, N, ref_mean_ms);

    res.kernel_ms = res.mean_ms;
    res.e2e_ms = res.h2d_ms + res.kernel_ms + res.d2h_ms;
    res.has_transfer_times = true;

    return res;
}
#elif defined(USE_CUDA) && !defined(__CUDACC__)
MethodResult run_cuda(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C_ref, int N, int warmups, int runs, double ref_mean_ms);
#else
static MethodResult run_cuda(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, int, int, int, double) {
    MethodResult res;
    res.name = "CUDA";
    res.available = false;
    res.device_info = "CUDA compiler support not enabled (-DUSE_CUDA nvcc)";
    return res;
}
#endif

// ============================================================================
// HARDWARE INFORMATION DISPLAY
// ============================================================================

static void display_system_information(int N) {
    int phys_cores = 0, log_cpus = 0;
    get_cpu_counts(phys_cores, log_cpus);
    
    std::string l1d, l1i, l2, l3;
    get_cache_info(l1d, l1i, l2, l3);
    
    std::string total_ram, avail_ram;
    get_ram_info(total_ram, avail_ram);

    std::cout << "============================================================\n";
    std::cout << "                 HARDWARE SPECIFICATIONS\n";
    std::cout << "============================================================\n";
    std::cout << "CPU DETAILS:\n";
    std::cout << "  Model              : " << get_cpu_model() << "\n";
    std::cout << "  Physical Cores     : " << phys_cores << "\n";
    std::cout << "  Logical Threads    : " << log_cpus << "\n";
    std::cout << "  RAM (Total / Free) : " << total_ram << " / " << avail_ram << "\n\n";

    std::cout << "CACHE MEMORY:\n";
    std::cout << "  L1 Data Cache      : " << l1d << "\n";
    std::cout << "  L1 Instruction     : " << l1i << "\n";
    std::cout << "  L2 Cache           : " << l2 << "\n";
    std::cout << "  L3 Cache           : " << l3 << "\n\n";

    std::cout << "GPU DETAILS:\n";
#ifdef USE_CUDA
    int cuda_dev_count = 0;
    if (cudaGetDeviceCount(&cuda_dev_count) == cudaSuccess && cuda_dev_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "  NVIDIA GPU         : " << prop.name << "\n";
        std::cout << "  GPU Memory (VRAM)  : " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
        std::cout << "  Compute Capability : " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Multiprocessors    : " << prop.multiProcessorCount << "\n";
    } else {
        std::cout << "  NVIDIA GPU         : None detected\n";
    }
#else
    std::cout << "  NVIDIA GPU         : Backend not enabled\n";
#endif

#ifdef USE_SYCL
    try {
        sycl::queue q{sycl::default_selector_v};
        sycl::device dev = q.get_device();
        sycl::platform plat = dev.get_platform();

        std::cout << "  SYCL Device        : " << dev.get_info<sycl::info::device::name>() << "\n";
        std::cout << "  SYCL Platform      : " << plat.get_info<sycl::info::platform::name>() << "\n";
        std::cout << "  SYCL Global Memory : " << (dev.get_info<sycl::info::device::global_mem_size>() / (1024 * 1024)) << " MB\n";
        std::cout << "  SYCL Compute Units : " << dev.get_info<sycl::info::device::max_compute_units>() << "\n";
    } catch (...) {
        std::cout << "  SYCL Device        : None detected\n";
    }
#else
    std::cout << "  SYCL Device        : Backend not enabled\n";
#endif
    std::cout << "============================================================\n\n";

    std::cout << "============================================================\n";
    std::cout << "BENCHMARK CONFIGURATION: Matrix " << N << " x " << N << " (float)\n";
    std::cout << "============================================================\n";
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [matrix_size] [omp_threads]\n";
    std::cout << "Options:\n";
    std::cout << "  matrix_size : Dimension N for N x N matrix (e.g. 512, 1024). Default: 1024\n";
    std::cout << "  omp_threads : Number of OpenMP threads to use. Default: max available\n";
    std::cout << "  --help      : Display this help message\n";
}

#ifndef MATMUL_NO_MAIN
int main(int argc, char** argv) {
    int N = 1024;
    int omp_threads = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (i == 1) {
            int val = std::atoi(arg.c_str());
            if (val > 0) N = val;
        } else if (i == 2) {
            int val = std::atoi(arg.c_str());
            if (val > 0) omp_threads = val;
        }
    }

    display_system_information(N);

    // Allocate & initialize matrices
    std::vector<float> A, B, C_ref;
    init_matrices(A, B, C_ref, N);

    const int WARMUPS = 5;
    const int RUNS = 10;

    std::cout << "Benchmarking implementations...\n";

    // 1. Normal C++ (CPU)
    std::cout << "  [CPU] Normal C++ (Single-threaded)... " << std::flush;
    MethodResult r_cpp = run_normal_cpp(A, B, C_ref, N, WARMUPS, RUNS);
    std::cout << "Done (" << std::fixed << std::setprecision(2) << r_cpp.mean_ms << " ms)\n";

    // 2. OpenMP (CPU)
    std::cout << "  [CPU] OpenMP (Multi-threaded)........ " << std::flush;
    MethodResult r_omp = run_openmp(A, B, C_ref, N, omp_threads, WARMUPS, RUNS, r_cpp.mean_ms);
    if (r_omp.available) {
        std::cout << "Done (" << std::fixed << std::setprecision(2) << r_omp.mean_ms << " ms)\n";
    } else {
        std::cout << "NOT AVAILABLE\n";
    }

    // 3. SYCL
    std::cout << "  [ACC] Intel SYCL..................... " << std::flush;
    MethodResult r_sycl = run_sycl(A, B, C_ref, N, WARMUPS, RUNS, r_cpp.mean_ms);
    if (r_sycl.available) {
        std::cout << "Done (" << std::fixed << std::setprecision(2) << r_sycl.mean_ms << " ms)\n";
    } else {
        std::cout << "NOT AVAILABLE (" << r_sycl.device_info << ")\n";
    }

    // 4. CUDA (GPU)
    std::cout << "  [GPU] NVIDIA CUDA.................... " << std::flush;
    MethodResult r_cuda = run_cuda(A, B, C_ref, N, WARMUPS, RUNS, r_cpp.mean_ms);
    if (r_cuda.available) {
        std::cout << "Done (" << std::fixed << std::setprecision(2) << r_cuda.mean_ms << " ms)\n";
    } else {
        std::cout << "NOT AVAILABLE (" << r_cuda.device_info << ")\n";
    }

    // ============================================================================
    // BENCHMARK RESULTS TABLE
    // ============================================================================

    std::vector<MethodResult> results = {r_cpp, r_omp, r_sycl, r_cuda};

    std::cout << "\n=======================================================================================\n";
    std::cout << "                          EXECUTION TIME & PERFORMANCE                                 \n";
    std::cout << "=======================================================================================\n";
    std::cout << std::left
              << std::setw(15) << "Backend"
              << std::setw(8)  << "Device"
              << std::setw(18) << "Execution Time"
              << std::setw(14) << "Min / Max (ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(10) << "Speedup"
              << std::setw(8)  << "Status"
              << "\n";
    std::cout << "---------------------------------------------------------------------------------------\n";

    for (const auto& res : results) {
        std::cout << std::left << std::setw(15) << res.name;
        std::string dev_label = (res.name == "CUDA") ? "GPU" : (res.name == "SYCL" ? "GPU/CPU" : "CPU");
        std::cout << std::left << std::setw(8) << dev_label;

        if (res.available) {
            std::ostringstream mean_ss, minmax_ss;
            mean_ss << std::fixed << std::setprecision(2) << res.mean_ms << " ms";
            minmax_ss << std::fixed << std::setprecision(1) << res.min_ms << " / " << res.max_ms;

            std::cout << std::left
                      << std::setw(18) << mean_ss.str()
                      << std::setw(14) << minmax_ss.str()
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << res.gflops;
            if (res.name == "Normal C++") {
                std::cout << std::setw(10) << "1.00x";
            } else if (res.verified) {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(2) << res.speedup << "x";
                std::cout << std::setw(10) << ss.str();
            } else {
                std::cout << std::setw(10) << "N/A";
            }
            std::cout << std::setw(8) << (res.verified ? "PASS" : "FAIL");
        } else {
            std::cout << std::setw(54) << "NOT AVAILABLE" << std::setw(8) << "N/A";
        }
        std::cout << "\n";
    }
    std::cout << "=======================================================================================\n";

    // Summary of CPU and GPU execution times
    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "EXECUTION TIME SUMMARY:\n";
    std::cout << "  CPU  (Normal C++) : " << std::fixed << std::setprecision(2) << r_cpp.mean_ms << " ms\n";
    if (r_omp.available) {
        std::cout << "  CPU  (OpenMP)     : " << std::fixed << std::setprecision(2) << r_omp.mean_ms << " ms\n";
    }
    if (r_sycl.available) {
        std::cout << "  SYCL (Accelerator): " << std::fixed << std::setprecision(2) << r_sycl.mean_ms << " ms\n";
    }
    if (r_cuda.available) {
        std::cout << "  GPU  (CUDA)       : " << std::fixed << std::setprecision(2) << r_cuda.mean_ms << " ms\n";
    }
    std::cout << "------------------------------------------------------------\n\n";

    return 0;
}
#endif
