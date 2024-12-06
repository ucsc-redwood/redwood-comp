#pragma once

#include <cuda_runtime.h>

#include "../utils.hpp"
#include "helpers.cuh"

namespace cuda {

template <typename T>
struct CudaAllocator {
  using value_type = T;

  T* allocate(std::size_t n) {
    SPD_TRACE_FUNC
    spdlog::debug("CUDA Allocator: Allocating {} bytes", n * sizeof(T));

    T* p;
    CUDA_CHECK(cudaMallocManaged(&p, n * sizeof(T)));
    return p;
  }

  void deallocate(T* p, std::size_t) {
    SPD_TRACE_FUNC
    CUDA_CHECK(cudaFree(p));
  }
};

}  // namespace cuda