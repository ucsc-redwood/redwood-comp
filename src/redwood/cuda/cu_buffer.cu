#include "cu_buffer.cuh"
#include "cu_helpers.cuh"

CudaBuffer::CudaBuffer(uint32_t size) : UsmBuffer(size) {
  CUDA_CHECK(cudaMallocManaged(&u_data, size));
}

CudaBuffer::~CudaBuffer() { CUDA_CHECK(cudaFree(u_data)); }
