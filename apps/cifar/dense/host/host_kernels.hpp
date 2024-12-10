#pragma once

#include <cstdint>

namespace cpu {

namespace kernels {

namespace dense {

// ----------------------------------------------------------------------------
// Convolution 2D (Dense)
// ----------------------------------------------------------------------------

void conv2d_mt(const float* input_data,
               const int image_input_channels,
               int input_height,
               int input_width,
               const float* weight_data,
               int weight_output_channels,
               int weight_input_channels,
               int weight_height,
               int weight_width,
               const float* bias_data,
               int bias_number_of_elements,
               int kernel_size,
               int stride,
               int padding,
               bool relu,
               float* output_data,
               // Threading parameters
               int start,
               int end);

// ----------------------------------------------------------------------------
// Max Pooling 2D (Dense)
// ----------------------------------------------------------------------------

void maxpool2d_mt(const float* input_data,
                  int input_channels,
                  int input_height,
                  int input_width,
                  int pool_size,
                  int stride,
                  float* output_data,
                  // Threading parameters
                  int start,
                  int end);

// ----------------------------------------------------------------------------
// Linear Layer (Dense)
// ----------------------------------------------------------------------------

void linear_mt(const float* input,
               const float* weights,
               const float* bias,
               float* output,
               uint32_t input_size,
               uint32_t output_size,
               // Threading parameters
               int start,
               int end);

}  // namespace dense

}  // namespace kernels

}  // namespace cpu
