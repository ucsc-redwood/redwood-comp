#pragma once

#include <cuda_runtime.h>

#include <memory_resource>

#include "../utils.hpp"
#include "helpers.cuh"

namespace cuda {

class CudaMemoryResource : public std::pmr::memory_resource {
 protected:
  void* do_allocate(std::size_t bytes, std::size_t) override {
    SPD_TRACE_FUNC

    void* ptr = nullptr;
    // cudaMallocManaged typically doesn't accept alignment directly,
    // but you can handle alignment manually if needed.
    CUDA_CHECK(cudaMallocManaged(&ptr, bytes));
    return ptr;
  }

  void do_deallocate(void* p, std::size_t, std::size_t) override {
    SPD_TRACE_FUNC

    CUDA_CHECK(cudaFree(p));
  }

  bool do_is_equal(const memory_resource& other) const noexcept override {
    // For simplicity, equality means same type.
    return dynamic_cast<const CudaMemoryResource*>(&other) != nullptr;
  }
};

}  // namespace cuda
