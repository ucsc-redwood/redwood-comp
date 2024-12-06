#include "device_kernels.cuh"

namespace cuda {

namespace kernels {

__global__ void vector_add(const int *input_a, const int *input_b, int *output,
                           size_t start, size_t end) {
  const auto i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= start && i < end) {
    output[i] = input_a[i] + input_b[i];
  }
}

} // namespace kernels

} // namespace cuda
