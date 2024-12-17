#pragma once

#include "../csr.hpp"

namespace cpu {

namespace kernels {

namespace sparse {

// ----------------------------------------------------------------------------
// Convolution 2D (Sparse)
// ----------------------------------------------------------------------------

// start, end = 0, weight_matrix.rows;
void conv2d_mt(const float* input_data,
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
               float* output_data,
               int start,
               int end);

// ----------------------------------------------------------------------------
// Max Pooling 2D (Dense)
// ----------------------------------------------------------------------------

// start, end = 0, input_channels * output_height * output_width
void maxpool2d_mt(const float* input_data,
                  int input_channels,
                  int input_height,
                  int input_width,
                  int pool_size,
                  int stride,
                  float* output_data,
                  int start,
                  int end);

// ----------------------------------------------------------------------------
// Linear Layer (Dense)
// ----------------------------------------------------------------------------

// start, end = 0, weight_matrix.rows
void linear_mt(const float* input_data,
               const CSRMatrix& weight_matrix,
               const float* bias_data,
               float* output_data,
               int start,
               int end);

}  // namespace sparse

}  // namespace kernels

}  // namespace cpu