// #include <cuda_runtime.h>

// #include "../utils.hpp"
// #include "cu_buffer.cuh"
// #include "helpers.cuh"

// namespace cuda {

// void Buffer::allocate() {
//   SPD_TRACE_FUNC;
//   CUDA_CHECK(cudaMallocManaged(&mapped_data_, size_));
// }

// void Buffer::free() {
//   SPD_TRACE_FUNC;
//   CUDA_CHECK(cudaFree(mapped_data_));
// }

// }  // namespace cuda
