#pragma once

#include <cstddef>

namespace cpu {

namespace kernels {

void vector_add(const int *input_a,
                const int *input_b,
                int *output,
                size_t start,
                size_t end);

}

}  // namespace cpu
