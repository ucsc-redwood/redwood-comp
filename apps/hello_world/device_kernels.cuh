#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda {

namespace kernels {

__global__ void vector_add(const int *input_a,
                           const int *input_b,
                           int *output,
                           size_t start,
                           size_t end);

}

}  // namespace cuda
