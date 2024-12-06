#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda {

namespace kernels {

__global__ void vector_add(const float *input_a,
                           const float *input_b,
                           float *output,
                           const size_t n);

}

}  // namespace cuda
