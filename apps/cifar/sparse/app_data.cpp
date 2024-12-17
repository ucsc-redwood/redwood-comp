#include "app_data.hpp"

#include <spdlog/spdlog.h>

#include <fstream>

#include "redwood/resources_path.hpp"

// void readDataFromFile(const std::string_view filename,
//                       float* data,
//                       const int size) {
//   const auto base_path = helpers::get_resource_base_path();

//   std::ifstream file(base_path / filename);
//   if (!file.is_open()) {
//     spdlog::error("Could not open file '{}'", (base_path /
//     filename).string()); return;
//   }

//   for (int i = 0; i < size; ++i) {
//     if (!(file >> data[i])) {
//       spdlog::error("Failed to read data at index {}", i);
//       return;
//     }
//   }
//   file.close();
// }

void readDataFromFile(const char* filename, float* data, int maxSize) {
  const auto base_path = helpers::get_resource_base_path();

  std::ifstream file(base_path / filename);
  if (!file.is_open()) {
    spdlog::error("Could not open the file - '{}'", filename);
    return;
  }

  // Zero initialize the entire array
  for (int i = 0; i < maxSize; ++i) {
    data[i] = 0.0f;
  }

  // Read available values
  float value;
  int count = 0;
  while (file >> value && count < maxSize) {
    data[count++] = value;
  }

  file.close();
}

void readCSRFromFiles(const char* values_file,
                      const char* row_ptr_file,
                      const char* col_idx_file,
                      float* values,
                      int* row_ptr,
                      int* col_idx,
                      int nnz,
                      int rows) {
  readDataFromFile(values_file, values, nnz);

  std::ifstream row_file(row_ptr_file);
  for (int i = 0; i <= rows; ++i) {
    row_file >> row_ptr[i];
  }
  row_file.close();

  std::ifstream col_file(col_idx_file);
  for (int i = 0; i < nnz; ++i) {
    col_file >> col_idx[i];
  }
  col_file.close();
}

AppData::AppData(std::pmr::memory_resource* mr)
    : BaseAppData(mr),
      // Image data
      u_image_data(3 * 32 * 32, mr),

      // Conv1 arrays
      u_conv1_values(MAX_NNZ_CONV1, mr),
      u_conv1_row_ptr(65, mr),  // 64 + 1
      u_conv1_col_idx(MAX_NNZ_CONV1, mr),

      // Conv2 arrays
      u_conv2_values(MAX_NNZ_CONV2, mr),
      u_conv2_row_ptr(193, mr),  // 192 + 1
      u_conv2_col_idx(MAX_NNZ_CONV2, mr),

      // Conv3 arrays
      u_conv3_values(MAX_NNZ_CONV3, mr),
      u_conv3_row_ptr(385, mr),  // 384 + 1
      u_conv3_col_idx(MAX_NNZ_CONV3, mr),

      // Conv4 arrays
      u_conv4_values(MAX_NNZ_CONV4, mr),
      u_conv4_row_ptr(257, mr),  // 256 + 1
      u_conv4_col_idx(MAX_NNZ_CONV4, mr),

      // Conv5 arrays
      u_conv5_values(MAX_NNZ_CONV5, mr),
      u_conv5_row_ptr(257, mr),  // 256 + 1
      u_conv5_col_idx(MAX_NNZ_CONV5, mr),

      // Linear arrays
      u_linear_values(MAX_NNZ_LINEAR, mr),
      u_linear_row_ptr(11, mr),  // 10 + 1
      u_linear_col_idx(MAX_NNZ_LINEAR, mr),

      // Intermediate results
      u_conv1_output(64 * 32 * 32, mr),
      u_pool1_output(64 * 16 * 16, mr),
      u_conv2_output(192 * 16 * 16, mr),
      u_pool2_output(192 * 8 * 8, mr),
      u_conv3_output(384 * 8 * 8, mr),
      u_conv4_output(256 * 8 * 8, mr),
      u_conv5_output(256 * 8 * 8, mr),
      u_pool3_output(256 * 4 * 4, mr),
      u_linear_output(10, mr),

      // Biases
      u_conv1_bias(64, mr),
      u_conv2_bias(192, mr),
      u_conv3_bias(384, mr),
      u_conv4_bias(256, mr),
      u_conv5_bias(256, mr),
      u_linear_bias(10, mr) {
  // TODO: Add code to load the sparse matrix data and biases from files
  // Similar to the dense version but with different file paths and formats

  // Load image data
  //   readDataFromFile(
  //       "images/flattened_bird_bird_57.txt", u_image_data.data(), 3072);
  readDataFromFile(
      "images/flattened_dog_dog_13.txt", u_image_data.data(), 3072);

  // Load CSR data for all layers
  readCSRFromFiles("sparse/conv1_values.txt",
                   "sparse/conv1_row_ptr.txt",
                   "sparse/conv1_col_idx.txt",
                   u_conv1_values.data(),
                   u_conv1_row_ptr.data(),
                   u_conv1_col_idx.data(),
                   MAX_NNZ_CONV1,
                   64);

  readCSRFromFiles("sparse/conv2_values.txt",
                   "sparse/conv2_row_ptr.txt",
                   "sparse/conv2_col_idx.txt",
                   u_conv2_values.data(),
                   u_conv2_row_ptr.data(),
                   u_conv2_col_idx.data(),
                   MAX_NNZ_CONV2,
                   192);

  readCSRFromFiles("sparse/conv3_values.txt",
                   "sparse/conv3_row_ptr.txt",
                   "sparse/conv3_col_idx.txt",
                   u_conv3_values.data(),
                   u_conv3_row_ptr.data(),
                   u_conv3_col_idx.data(),
                   MAX_NNZ_CONV3,
                   384);

  readCSRFromFiles("sparse/conv4_values.txt",
                   "sparse/conv4_row_ptr.txt",
                   "sparse/conv4_col_idx.txt",
                   u_conv4_values.data(),
                   u_conv4_row_ptr.data(),
                   u_conv4_col_idx.data(),
                   MAX_NNZ_CONV4,
                   256);

  readCSRFromFiles("sparse/conv5_values.txt",
                   "sparse/conv5_row_ptr.txt",
                   "sparse/conv5_col_idx.txt",
                   u_conv5_values.data(),
                   u_conv5_row_ptr.data(),
                   u_conv5_col_idx.data(),
                   MAX_NNZ_CONV5,
                   256);

  readCSRFromFiles("sparse/linear_values.txt",
                   "sparse/linear_row_ptr.txt",
                   "sparse/linear_col_idx.txt",
                   u_linear_values.data(),
                   u_linear_row_ptr.data(),
                   u_linear_col_idx.data(),
                   MAX_NNZ_LINEAR,
                   10);

  // Load biases
  readDataFromFile("sparse/conv1_bias.txt", u_conv1_bias.data(), 64);
  readDataFromFile("sparse/conv2_bias.txt", u_conv2_bias.data(), 192);
  readDataFromFile("sparse/conv3_bias.txt", u_conv3_bias.data(), 384);
  readDataFromFile("sparse/conv4_bias.txt", u_conv4_bias.data(), 256);
  readDataFromFile("sparse/conv5_bias.txt", u_conv5_bias.data(), 256);
  readDataFromFile("sparse/linear_bias.txt", u_linear_bias.data(), 10);

  // Create CSR matrices
  conv1_weights = {u_conv1_values.data(),
                   u_conv1_row_ptr.data(),
                   u_conv1_col_idx.data(),
                   64,
                   27,
                   MAX_NNZ_CONV1};
  conv2_weights = {u_conv2_values.data(),
                   u_conv2_row_ptr.data(),
                   u_conv2_col_idx.data(),
                   192,
                   576,
                   MAX_NNZ_CONV2};
  conv3_weights = {u_conv3_values.data(),
                   u_conv3_row_ptr.data(),
                   u_conv3_col_idx.data(),
                   384,
                   1728,
                   MAX_NNZ_CONV3};
  conv4_weights = {u_conv4_values.data(),
                   u_conv4_row_ptr.data(),
                   u_conv4_col_idx.data(),
                   256,
                   3456,
                   MAX_NNZ_CONV4};
  conv5_weights = {u_conv5_values.data(),
                   u_conv5_row_ptr.data(),
                   u_conv5_col_idx.data(),
                   256,
                   2304,
                   MAX_NNZ_CONV5};
  linear_weights = {u_linear_values.data(),
                    u_linear_row_ptr.data(),
                    u_linear_col_idx.data(),
                    10,
                    4096,
                    MAX_NNZ_LINEAR};

  // Create CSR matrices

  //   CSRMatrix conv1_weights(mr);
  //   conv1_weights.values = u_conv1_values;
  //   conv1_weights.row_ptr = u_conv1_row_ptr;
  //   conv1_weights.col_idx = u_conv1_col_idx;
  //   conv1_weights.rows = 64;
  //   conv1_weights.cols = 27;
  //   conv1_weights.nnz = MAX_NNZ_CONV1;

  //   CSRMatrix conv2_weights(mr);
  //   conv2_weights.values = u_conv2_values;
  //   conv2_weights.row_ptr = u_conv2_row_ptr;
  //   conv2_weights.col_idx = u_conv2_col_idx;
  //   conv2_weights.rows = 192;
  //   conv2_weights.cols = 576;
  //   conv2_weights.nnz = MAX_NNZ_CONV2;

  //   CSRMatrix conv3_weights(mr);
  //   conv3_weights.values = u_conv3_values;
  //   conv3_weights.row_ptr = u_conv3_row_ptr;
  //   conv3_weights.col_idx = u_conv3_col_idx;
  //   conv3_weights.rows = 384;
  //   conv3_weights.cols = 1728;
  //   conv3_weights.nnz = MAX_NNZ_CONV3;

  //   CSRMatrix conv4_weights(mr);
  //   conv4_weights.values = u_conv4_values;
  //   conv4_weights.row_ptr = u_conv4_row_ptr;
  //   conv4_weights.col_idx = u_conv4_col_idx;
  //   conv4_weights.rows = 256;
  //   conv4_weights.cols = 3456;
  //   conv4_weights.nnz = MAX_NNZ_CONV4;

  //   CSRMatrix conv5_weights(mr);
  //   conv5_weights.values = u_conv5_values;
  //   conv5_weights.row_ptr = u_conv5_row_ptr;
  //   conv5_weights.col_idx = u_conv5_col_idx;
  //   conv5_weights.rows = 256;
  //   conv5_weights.cols = 2304;
  //   conv5_weights.nnz = MAX_NNZ_CONV5;

  //   CSRMatrix linear_weights(mr);
  //   linear_weights.values = u_linear_values;
  //   linear_weights.row_ptr = u_linear_row_ptr;
  //   linear_weights.col_idx = u_linear_col_idx;
  //   linear_weights.rows = 10;
  //   linear_weights.cols = 4096;
  //   linear_weights.nnz = MAX_NNZ_LINEAR;
}
