#pragma once

#include "redwood/base_appdata.hpp"

// ----------------------------------------------------------------------------
// Model Architecture Parameters
// ----------------------------------------------------------------------------

namespace model {

constexpr int input_channels = 3;
constexpr int input_height = 32;
constexpr int input_width = 32;
constexpr int kernel_size = 3;
constexpr int stride = 1;
constexpr int padding = 1;
constexpr int pool_size = 2;
constexpr int pool_stride = 2;
constexpr bool use_relu = true;
constexpr float sparse_threshold = 1e-6;

// Network layer dimensions
constexpr int conv1_filters = 64;
constexpr int conv2_filters = 192;
constexpr int conv3_filters = 384;
constexpr int conv4_filters = 256;
constexpr int conv5_filters = 256;
constexpr int num_classes = 10;

}  // namespace model

// ----------------------------------------------------------------------------
// Output Dimensions
// ----------------------------------------------------------------------------

namespace dims {

constexpr auto conv_output = [](int input_dim) {
  return (input_dim + 2 * model::padding - model::kernel_size) / model::stride +
         1;
};
constexpr auto pool_output = [](int input_dim) {
  return (input_dim - model::pool_size) / model::pool_stride + 1;
};

constexpr int conv1_h = conv_output(model::input_height);  // 32
constexpr int conv1_w = conv_output(model::input_width);   // 32
constexpr int pool1_h = pool_output(conv1_h);              // 16
constexpr int pool1_w = pool_output(conv1_w);              // 16
constexpr int pool2_h = pool_output(pool1_h);              // 8
constexpr int pool2_w = pool_output(pool1_w);              // 8
constexpr int pool3_h = pool_output(pool2_h);              // 4
constexpr int pool3_w = pool_output(pool2_w);              // 4
constexpr int flattened_size =
    model::conv5_filters * pool3_h * pool3_w;  // 4096

}  // namespace dims

// ----------------------------------------------------------------------------
// Data (Buffers)
// ----------------------------------------------------------------------------

struct AppData final : public BaseAppData {
  explicit AppData(std::pmr::memory_resource* mr);

  UsmVector<float> u_image;

  // weights and biases
  UsmVector<float> u_conv1_weights;
  UsmVector<float> u_conv1_bias;
  UsmVector<float> u_conv2_weights;
  UsmVector<float> u_conv2_bias;
  UsmVector<float> u_conv3_weights;
  UsmVector<float> u_conv3_bias;
  UsmVector<float> u_conv4_weights;
  UsmVector<float> u_conv4_bias;
  UsmVector<float> u_conv5_weights;
  UsmVector<float> u_conv5_bias;
  UsmVector<float> u_linear_weights;
  UsmVector<float> u_linear_bias;
  UsmVector<float> u_flattened_output;
  UsmVector<float> u_final_output;

  // outputs?
  UsmVector<float> u_conv1_output;
  UsmVector<float> u_pool1_output;
  UsmVector<float> u_conv2_output;
  UsmVector<float> u_pool2_output;
  UsmVector<float> u_conv3_output;
  UsmVector<float> u_conv4_output;
  UsmVector<float> u_conv5_output;
  UsmVector<float> u_pool3_output;
};