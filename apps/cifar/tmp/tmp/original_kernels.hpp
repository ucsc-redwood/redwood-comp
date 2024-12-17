#pragma once

#include "../csr.hpp"

namespace cpu::sparse::kernels {

void sparseConv2d(const v1::CSRMatrix& sparse_input,
                  int input_height,
                  int input_width,
                  const v1::CSRMatrix& sparse_weights,
                  int weight_output_channels,
                  float* bias_data,
                  int kernel_size,
                  int stride,
                  int padding,
                  bool relu,
                  float* output_data);

void sparseMaxPool2d(const v1::CSRMatrix& sparse_input,
                     int input_channels,
                     int input_height,
                     int input_width,
                     int pool_size,
                     int stride,
                     float* output_data);

void sparseLinearLayer(const v1::CSRMatrix& sparse_input,
                       const v1::CSRMatrix& sparse_weights,
                       float* bias_data,
                       float* output_data,
                       int output_size);

}  // namespace cpu::sparse::kernels
