cmake_minimum_required(VERSION 3.18)
project(UnifiedMatrixMultiplicationBenchmark LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(ENABLE_OPENMP "Enable OpenMP support" ON)
option(ENABLE_SYCL   "Enable SYCL support (needs CMAKE_CXX_COMPILER=icpx)" ON)
option(ENABLE_CUDA   "Enable CUDA support (needs nvcc)" ON)

set(BENCHMARK_SRCS unified_matrix_benchmark.cpp)

# ---------------- OpenMP ----------------
if(ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - OpenMP backend enabled")
    add_compile_definitions(USE_OPENMP)
  else()
    message(STATUS "OpenMP NOT found")
  endif()
endif()

# ---------------- SYCL ----------------
if(ENABLE_SYCL)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message(STATUS "IntelLLVM compiler detected - SYCL backend enabled")
    add_compile_definitions(USE_SYCL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
  else()
    message(STATUS "C++ compiler is ${CMAKE_CXX_COMPILER_ID}, not icpx - SYCL backend disabled")
  endif()
endif()

# ---------------- CUDA ----------------
if(ENABLE_CUDA)
  include(CheckLanguage)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_VERSION VERSION_LESS 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER} - CUDA backend enabled")
    add_compile_definitions(USE_CUDA)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
      set_source_files_properties(unified_matrix_benchmark.cpp PROPERTIES LANGUAGE CUDA)
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -O3")
  else()
    message(STATUS "nvcc NOT found - CUDA backend disabled")
  endif()
endif()

add_executable(unified_matrix_benchmark ${BENCHMARK_SRCS})

if(EXISTS "/opt/intel/oneapi/compiler/latest/include")
  target_include_directories(unified_matrix_benchmark PRIVATE /opt/intel/oneapi/compiler/latest/include)
elseif(EXISTS "/opt/intel/oneapi/compiler/2026.1/include")
  target_include_directories(unified_matrix_benchmark PRIVATE /opt/intel/oneapi/compiler/2026.1/include)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(unified_matrix_benchmark PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CUDA_COMPILER)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(unified_matrix_benchmark PRIVATE CUDA::cudart)
endif()
