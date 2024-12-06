#include "host_kernels.hpp"

namespace cpu {

namespace kernels {

void vector_add(const float *input_a,
                const float *input_b,
                float *output,
                const size_t start,
                const size_t end) {
  for (size_t i = start; i < end; ++i) {
    output[i] = input_a[i] + input_b[i];
  }
}

}  // namespace kernels

}  // namespace cpu
