#pragma once

#include "csr.hpp"
#include "redwood/base_appdata.hpp"

// Maximum sizes for static arrays
constexpr int MAX_NNZ_CONV1 = 1728;    // 3*3*3*64
constexpr int MAX_NNZ_CONV2 = 110592;  // 3*3*64*192
constexpr int MAX_NNZ_CONV3 = 663552;  // 3*3*192*384
constexpr int MAX_NNZ_CONV4 = 884736;  // 3*3*384*256
constexpr int MAX_NNZ_CONV5 = 589824;  // 3*3*256*256
constexpr int MAX_NNZ_LINEAR = 40960;  // 256*4*4*10

struct AppData : public BaseAppData {
  // // Static arrays for all layers
  // float conv1_values[MAX_NNZ_CONV1];
  // int conv1_row_ptr[65];  // 64 + 1
  // int conv1_col_idx[MAX_NNZ_CONV1];

  // float conv2_values[MAX_NNZ_CONV2];
  // int conv2_row_ptr[193];  // 192 + 1
  // int conv2_col_idx[MAX_NNZ_CONV2];

  // float conv3_values[MAX_NNZ_CONV3];
  // int conv3_row_ptr[385];  // 384 + 1
  // int conv3_col_idx[MAX_NNZ_CONV3];

  // float conv4_values[MAX_NNZ_CONV4];
  // int conv4_row_ptr[257];  // 256 + 1
  // int conv4_col_idx[MAX_NNZ_CONV4];

  // float conv5_values[MAX_NNZ_CONV5];
  // int conv5_row_ptr[257];  // 256 + 1
  // int conv5_col_idx[MAX_NNZ_CONV5];

  // float linear_values[MAX_NNZ_LINEAR];
  // int linear_row_ptr[11];  // 10 + 1
  // int linear_col_idx[MAX_NNZ_LINEAR];

  // // Static arrays for intermediate results
  // float conv1_output[64 * 32 * 32];
  // float pool1_output[64 * 16 * 16];
  // float conv2_output[192 * 16 * 16];
  // float pool2_output[192 * 8 * 8];
  // float conv3_output[384 * 8 * 8];
  // float conv4_output[256 * 8 * 8];
  // float conv5_output[256 * 8 * 8];
  // float pool3_output[256 * 4 * 4];
  // float linear_output[10];

  // // Static arrays for biases
  // float conv1_bias[64];
  // float conv2_bias[192];
  // float conv3_bias[384];
  // float conv4_bias[256];
  // float conv5_bias[256];
  // float linear_bias[10];

  UsmVector<float> u_image_data;

  UsmVector<float> u_conv1_values;
  UsmVector<int> u_conv1_row_ptr;
  UsmVector<int> u_conv1_col_idx;

  UsmVector<float> u_conv2_values;
  UsmVector<int> u_conv2_row_ptr;
  UsmVector<int> u_conv2_col_idx;

  UsmVector<float> u_conv3_values;
  UsmVector<int> u_conv3_row_ptr;
  UsmVector<int> u_conv3_col_idx;

  UsmVector<float> u_conv4_values;
  UsmVector<int> u_conv4_row_ptr;
  UsmVector<int> u_conv4_col_idx;

  UsmVector<float> u_conv5_values;
  UsmVector<int> u_conv5_row_ptr;
  UsmVector<int> u_conv5_col_idx;

  UsmVector<float> u_linear_values;
  UsmVector<int> u_linear_row_ptr;
  UsmVector<int> u_linear_col_idx;

  UsmVector<float> u_conv1_output;
  UsmVector<float> u_pool1_output;
  UsmVector<float> u_conv2_output;
  UsmVector<float> u_pool2_output;
  UsmVector<float> u_conv3_output;
  UsmVector<float> u_conv4_output;
  UsmVector<float> u_conv5_output;
  UsmVector<float> u_pool3_output;
  UsmVector<float> u_linear_output;

  UsmVector<float> u_conv1_bias;
  UsmVector<float> u_conv2_bias;
  UsmVector<float> u_conv3_bias;
  UsmVector<float> u_conv4_bias;
  UsmVector<float> u_conv5_bias;
  UsmVector<float> u_linear_bias;

  CSRMatrix conv1_weights;
  CSRMatrix conv2_weights;
  CSRMatrix conv3_weights;
  CSRMatrix conv4_weights;
  CSRMatrix conv5_weights;
  CSRMatrix linear_weights;

  explicit AppData(std::pmr::memory_resource* mr);
};