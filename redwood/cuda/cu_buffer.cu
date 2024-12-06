#include "cu_buffer.cuh"

#include "../utils.hpp"

#include <cuda_runtime.h>

namespace cuda {

void Buffer::allocate() {
  SPD_TRACE_FUNC;
  cudaMallocManaged(&mapped_data_, size_);
}

void Buffer::free() {
  SPD_TRACE_FUNC;
  cudaFree(mapped_data_);
}

} // namespace cuda
