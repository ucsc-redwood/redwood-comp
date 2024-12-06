#include "device_kernels.cuh"

namespace cuda {

namespace kernels {

__global__ void vector_add(const float *input_a,
                           const float *input_b,
                           float *output,
                           const size_t n) {
  const auto i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    output[i] = input_a[i] + input_b[i];
  }
}

}  // namespace kernels

}  // namespace cuda
