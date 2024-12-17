#include "original_kernels.hpp"

#include <cfloat>

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
                  float* output_data) {
  int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
  int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
  int output_size = weight_output_channels * output_height * output_width;

  std::fill(output_data, output_data + output_size, 0.0f);

  struct PaddedPosition {
    int x, y, channel;
    float value;
  };

  std::vector<std::vector<PaddedPosition>> padded_positions(
      (input_height + 2 * padding) * (input_width + 2 * padding));

  // Populate padded positions
  for (int in_c = 0; in_c < sparse_input.rows; ++in_c) {
    for (int idx = sparse_input.row_ptr[in_c];
         idx < sparse_input.row_ptr[in_c + 1];
         ++idx) {
      int orig_pos = sparse_input.col_indices[idx];
      int orig_y = orig_pos / input_width;
      int orig_x = orig_pos % input_width;

      int padded_y = orig_y + padding;
      int padded_x = orig_x + padding;

      padded_positions[padded_y * (input_width + 2 * padding) + padded_x]
          .push_back({padded_x, padded_y, in_c, sparse_input.values[idx]});
    }
  }

  // Pre-compute weight indices
  std::vector<std::vector<std::pair<int, float>>> weight_values(
      weight_output_channels);
  for (int out_c = 0; out_c < weight_output_channels; ++out_c) {
    for (int w_idx = sparse_weights.row_ptr[out_c];
         w_idx < sparse_weights.row_ptr[out_c + 1];
         ++w_idx) {
      weight_values[out_c].push_back(
          {sparse_weights.col_indices[w_idx], sparse_weights.values[w_idx]});
    }
  }

  // Main convolution loop
  for (int out_c = 0; out_c < weight_output_channels; ++out_c) {
    float bias = bias_data ? bias_data[out_c] : 0.0f;
    const auto& channel_weights = weight_values[out_c];

    for (int h = 0; h < output_height; ++h) {
      for (int w = 0; w < output_width; ++w) {
        float sum = bias;
        int output_idx = (out_c * output_height + h) * output_width + w;

        for (const auto& weight_pair : channel_weights) {
          int weight_offset = weight_pair.first;
          int in_c = weight_offset / (kernel_size * kernel_size);
          int ky = (weight_offset % (kernel_size * kernel_size)) / kernel_size;
          int kx = weight_offset % kernel_size;

          int in_y = h * stride + ky;
          int in_x = w * stride + kx;

          const auto& positions =
              padded_positions[in_y * (input_width + 2 * padding) + in_x];
          for (const auto& pos : positions) {
            if (pos.channel == in_c) {
              sum += pos.value * weight_pair.second;
              break;
            }
          }
        }

        output_data[output_idx] = relu && sum < 0 ? 0.0f : sum;
      }
    }
  }
}

void sparseMaxPool2d(const v1::CSRMatrix& sparse_input,
                     int input_channels,
                     int input_height,
                     int input_width,
                     int pool_size,
                     int stride,
                     float* output_data) {
  int output_height = (input_height - pool_size) / stride + 1;
  int output_width = (input_width - pool_size) / stride + 1;
  int output_size = input_channels * output_height * output_width;

  std::fill(output_data, output_data + output_size, -FLT_MAX);

  struct OutputPosition {
    int x, y;
    float value;
  };

  std::vector<std::vector<OutputPosition>> output_positions(input_channels);

  // Pre-compute output positions
  for (int c = 0; c < input_channels; ++c) {
    for (int row = sparse_input.row_ptr[c]; row < sparse_input.row_ptr[c + 1];
         ++row) {
      int col = sparse_input.col_indices[row];
      int in_x = col % input_width;
      int in_y = (col / input_width) % input_height;
      float value = sparse_input.values[row];

      int out_start_x = std::max(0, (in_x - pool_size + stride) / stride);
      int out_start_y = std::max(0, (in_y - pool_size + stride) / stride);
      int out_end_x = std::min(output_width - 1, in_x / stride);
      int out_end_y = std::min(output_height - 1, in_y / stride);

      for (int out_y = out_start_y; out_y <= out_end_y; ++out_y) {
        for (int out_x = out_start_x; out_x <= out_end_x; ++out_x) {
          if (in_x >= out_x * stride && in_x < out_x * stride + pool_size &&
              in_y >= out_y * stride && in_y < out_y * stride + pool_size) {
            output_positions[c].push_back({out_x, out_y, value});
          }
        }
      }
    }
  }

  // Update output values
  for (int c = 0; c < input_channels; ++c) {
    for (const auto& pos : output_positions[c]) {
      int out_idx = (c * output_height + pos.y) * output_width + pos.x;
      output_data[out_idx] = std::max(output_data[out_idx], pos.value);
    }
  }

  // Replace -FLT_MAX with zeros
  for (int i = 0; i < output_size; ++i) {
    if (output_data[i] == -FLT_MAX) {
      output_data[i] = 0.0f;
    }
  }
}

void sparseLinearLayer(const v1::CSRMatrix& sparse_input,
                       const v1::CSRMatrix& sparse_weights,
                       float* bias,
                       float* output_data,
                       int output_size) {
  std::vector<std::pair<int, float>> input_values;
  for (int in_idx = sparse_input.row_ptr[0]; in_idx < sparse_input.row_ptr[1];
       ++in_idx) {
    input_values.push_back(
        {sparse_input.col_indices[in_idx], sparse_input.values[in_idx]});
  }

  // Initialize with bias
  for (int i = 0; i < output_size; ++i) {
    output_data[i] = bias[i];
  }

  // Process weights
  for (int i = 0; i < output_size; ++i) {
    const auto& row_start = sparse_weights.row_ptr[i];
    const auto& row_end = sparse_weights.row_ptr[i + 1];

    for (const auto& input_pair : input_values) {
      auto weight_it =
          std::lower_bound(sparse_weights.col_indices.begin() + row_start,
                           sparse_weights.col_indices.begin() + row_end,
                           input_pair.first);

      if (weight_it != sparse_weights.col_indices.begin() + row_end &&
          *weight_it == input_pair.first) {
        int weight_idx = weight_it - sparse_weights.col_indices.begin();
        output_data[i] += input_pair.second * sparse_weights.values[weight_idx];
      }
    }
  }
}

}  // namespace cpu::sparse::kernels