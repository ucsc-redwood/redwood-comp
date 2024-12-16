#pragma once

#include "../csr.hpp"

namespace cpu {

namespace kernels {

namespace sparse {

// ----------------------------------------------------------------------------
// Convolution 2D (Sparse)
// ----------------------------------------------------------------------------

void conv2d(const float* input_data,
            int image_input_channels,
            int input_height,
            int input_width,
            const CSRMatrix& weight_matrix,
            const float* bias_data,
            int bias_size,
            int kernel_size,
            int stride,
            int padding,
            bool relu,
            float* output_data);

// ----------------------------------------------------------------------------
// Max Pooling 2D (Dense)
// ----------------------------------------------------------------------------
void maxpool2d(const float* input_data,
               int input_channels,
               int input_height,
               int input_width,
               int pool_size,
               int stride,
               float* output_data);

// ----------------------------------------------------------------------------
// Linear Layer (Dense)
// ----------------------------------------------------------------------------
void linear(const float* input_data,
            const CSRMatrix& weight_matrix,
            const float* bias_data,
            float* output_data);

}  // namespace sparse

}  // namespace kernels

}  // namespace cpu