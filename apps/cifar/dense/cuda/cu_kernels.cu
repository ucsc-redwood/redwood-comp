#include "cu_kernels.cuh"

namespace cuda {

namespace kernels {

namespace dense {

__global__ void conv2d(const float* __restrict__ input_data,
                       const float* __restrict__ weight_data,
                       const float* __restrict__ bias_data,
                       float* __restrict__ output_data,
                       const int input_height,
                       const int input_width,
                       const int weight_output_channels,
                       const int weight_input_channels,
                       const int weight_height,
                       const int weight_width,
                       const int bias_number_of_elements,
                       const int kernel_size,
                       const int stride,
                       const int padding,
                       const int output_height,
                       const int output_width,
                       const bool relu) {
  // Compute global index (linear index within the output buffer)
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Total number of output elements
  int total_output_elements =
      weight_output_channels * output_height * output_width;

  // Check that the thread is within the valid range
  if (global_idx >= total_output_elements) return;

  // Compute output coordinates: (out_channel, y, x)
  int out_channel = global_idx / (output_height * output_width);
  int hw_idx = global_idx % (output_height * output_width);
  int y = hw_idx / output_width;
  int x = hw_idx % output_width;

  float sum = 0.0f;

  // Perform the convolution
  for (int in_channel = 0; in_channel < weight_input_channels; ++in_channel) {
    for (int ky = 0; ky < weight_height; ++ky) {
      int image_y_base = y * stride + ky - padding;
      for (int kx = 0; kx < weight_width; ++kx) {
        int image_x = x * stride + kx - padding;

        // Check boundaries
        if (image_y_base >= 0 && image_y_base < input_height && image_x >= 0 &&
            image_x < input_width) {
          int input_index =
              ((in_channel * input_height + image_y_base) * input_width +
               image_x);
          int weight_index =
              ((((out_channel * weight_input_channels) + in_channel) *
                    weight_height +
                ky) *
                   weight_width +
               kx);

          sum += input_data[input_index] * weight_data[weight_index];
        }
      }
    }
  }

  // Add bias
  if (bias_data && out_channel < bias_number_of_elements) {
    sum += bias_data[out_channel];
  }

  // Apply ReLU if enabled
  if (relu && sum < 0.0f) {
    sum = 0.0f;
  }

  // Store the result
  int output_index = (out_channel * output_height + y) * output_width + x;
  output_data[output_index] = sum;
}

}  // namespace dense

}  // namespace kernels

}  // namespace cuda
