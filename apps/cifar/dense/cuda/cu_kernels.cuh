#pragma once

#include <cuda_runtime.h>

namespace cuda {

namespace kernels {

namespace dense {

__global__ void conv2d(const float* __restrict__ input_data,
                       const float* __restrict__ weight_data,
                       const float* __restrict__ bias_data,
                       float* __restrict__ output_data,
                       int input_height,
                       int input_width,
                       int weight_output_channels,
                       int weight_input_channels,
                       int weight_height,
                       int weight_width,
                       int bias_number_of_elements,
                       int kernel_size,
                       int stride,
                       int padding,
                       int output_height,
                       int output_width,
                       bool relu);

}  // namespace dense

}  // namespace kernels

}  // namespace cuda
