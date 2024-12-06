#pragma once

#include <cstddef>

namespace cpu {

namespace kernels {

void vector_add(const float *input_a,
                const float *input_b,
                float *output,
                size_t start,
                size_t end);

}

}  // namespace cpu
