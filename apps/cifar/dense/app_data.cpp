#include "app_data.hpp"

#include <spdlog/spdlog.h>

#include <fstream>

#include "redwood/resources_path.hpp"

void readDataFromFile(const std::string_view filename,
                      float* data,
                      const int size) {
  const auto base_path = helpers::get_resource_base_path();

  std::ifstream file(base_path / filename);
  if (!file.is_open()) {
    spdlog::error("Could not open file '{}'", (base_path / filename).string());
    return;
  }

  for (int i = 0; i < size; ++i) {
    if (!(file >> data[i])) {
      spdlog::error("Failed to read data at index {}", i);
      return;
    }
  }
  file.close();
}

// v1::Data::Data(Engine& engine)
//     : u_image(engine.typed_buffer<float>(
//           model::input_channels * model::input_height * model::input_width)),
//       u_conv1_weights(engine.typed_buffer<float>(
//           model::conv1_filters * model::input_channels * model::kernel_size *
//           model::kernel_size)),
//       u_conv1_bias(engine.typed_buffer<float>(model::conv1_filters)),
//       u_conv2_weights(engine.typed_buffer<float>(
//           model::conv2_filters * model::conv1_filters * model::kernel_size *
//           model::kernel_size)),
//       u_conv2_bias(engine.typed_buffer<float>(model::conv2_filters)),
//       u_conv3_weights(engine.typed_buffer<float>(
//           model::conv3_filters * model::conv2_filters * model::kernel_size *
//           model::kernel_size)),
//       u_conv3_bias(engine.typed_buffer<float>(model::conv3_filters)),
//       u_conv4_weights(engine.typed_buffer<float>(
//           model::conv4_filters * model::conv3_filters * model::kernel_size *
//           model::kernel_size)),
//       u_conv4_bias(engine.typed_buffer<float>(model::conv4_filters)),
//       u_conv5_weights(engine.typed_buffer<float>(
//           model::conv5_filters * model::conv4_filters * model::kernel_size *
//           model::kernel_size)),
//       u_conv5_bias(engine.typed_buffer<float>(model::conv5_filters)),
//       u_linear_weights(engine.typed_buffer<float>(model::num_classes *
//                                                   dims::flattened_size)),
//       u_linear_bias(engine.typed_buffer<float>(model::num_classes)),
//       u_flattened_output(engine.typed_buffer<float>(dims::flattened_size)),
//       u_final_output(engine.typed_buffer<float>(model::num_classes)),
//       u_conv1_output(engine.typed_buffer<float>(model::conv1_filters *
//                                                 dims::conv1_h *
//                                                 dims::conv1_w)),
//       u_pool1_output(engine.typed_buffer<float>(model::conv1_filters *
//                                                 dims::pool1_h *
//                                                 dims::pool1_w)),
//       u_conv2_output(engine.typed_buffer<float>(model::conv2_filters *
//                                                 dims::pool1_h *
//                                                 dims::pool1_w)),
//       u_pool2_output(engine.typed_buffer<float>(model::conv2_filters *
//                                                 dims::pool2_h *
//                                                 dims::pool2_w)),
//       u_conv3_output(engine.typed_buffer<float>(model::conv3_filters *
//                                                 dims::pool2_h *
//                                                 dims::pool2_w)),
//       u_conv4_output(engine.typed_buffer<float>(model::conv4_filters *
//                                                 dims::pool2_h *
//                                                 dims::pool2_w)),
//       u_conv5_output(engine.typed_buffer<float>(model::conv5_filters *
//                                                 dims::pool2_h *
//                                                 dims::pool2_w)),
//       u_pool3_output(engine.typed_buffer<float>(
//           model::conv5_filters * dims::pool3_h * dims::pool3_w)) {

AppData::AppData(std::pmr::memory_resource* mr)
    : BaseAppData(mr),
      u_image(model::input_channels * model::input_height * model::input_width,
              mr),
      u_conv1_weights(model::conv1_filters * model::input_channels *
                          model::kernel_size * model::kernel_size,
                      mr),
      u_conv1_bias(model::conv1_filters, mr),
      u_conv2_weights(model::conv2_filters * model::conv1_filters *
                          model::kernel_size * model::kernel_size,
                      mr),
      u_conv2_bias(model::conv2_filters, mr),
      u_conv3_weights(model::conv3_filters * model::conv2_filters *
                          model::kernel_size * model::kernel_size,
                      mr),
      u_conv3_bias(model::conv3_filters, mr),
      u_conv4_weights(model::conv4_filters * model::conv3_filters *
                          model::kernel_size * model::kernel_size,
                      mr),
      u_conv4_bias(model::conv4_filters, mr),
      u_conv5_weights(model::conv5_filters * model::conv4_filters *
                          model::kernel_size * model::kernel_size,
                      mr),
      u_conv5_bias(model::conv5_filters, mr),
      u_linear_weights(model::num_classes * dims::flattened_size, mr),
      u_linear_bias(model::num_classes, mr),
      u_flattened_output(dims::flattened_size, mr),
      u_final_output(model::num_classes, mr),
      u_conv1_output(model::conv1_filters * dims::conv1_h * dims::conv1_w, mr),
      u_pool1_output(model::conv1_filters * dims::pool1_h * dims::pool1_w, mr),
      u_conv2_output(model::conv2_filters * dims::pool1_h * dims::pool1_w, mr),
      u_pool2_output(model::conv2_filters * dims::pool2_h * dims::pool2_w, mr),
      u_conv3_output(model::conv3_filters * dims::pool2_h * dims::pool2_w, mr),
      u_conv4_output(model::conv4_filters * dims::pool2_h * dims::pool2_w, mr),
      u_conv5_output(model::conv5_filters * dims::pool2_h * dims::pool2_w, mr),
      u_pool3_output(model::conv5_filters * dims::pool3_h * dims::pool3_w, mr) {
  // Load input image
  readDataFromFile(
      "images/flattened_bird_bird_57.txt",
      u_image.data(),
      model::input_channels * model::input_height * model::input_width);

  // Load conv1 parameters
  readDataFromFile("sparse/features_0_weight.txt",
                   u_conv1_weights.data(),
                   model::conv1_filters * model::input_channels *
                       model::kernel_size * model::kernel_size);
  readDataFromFile(
      "sparse/features_0_bias.txt", u_conv1_bias.data(), model::conv1_filters);

  // Load conv2 parameters
  readDataFromFile("sparse/features_3_weight.txt",
                   u_conv2_weights.data(),
                   model::conv2_filters * model::conv1_filters *
                       model::kernel_size * model::kernel_size);
  readDataFromFile(
      "sparse/features_3_bias.txt", u_conv2_bias.data(), model::conv2_filters);

  // Load conv3 parameters
  readDataFromFile("sparse/features_6_weight.txt",
                   u_conv3_weights.data(),
                   model::conv3_filters * model::conv2_filters *
                       model::kernel_size * model::kernel_size);
  readDataFromFile(
      "sparse/features_6_bias.txt", u_conv3_bias.data(), model::conv3_filters);

  // Load conv4 parameters
  readDataFromFile("sparse/features_8_weight.txt",
                   u_conv4_weights.data(),
                   model::conv4_filters * model::conv3_filters *
                       model::kernel_size * model::kernel_size);
  readDataFromFile(
      "sparse/features_8_bias.txt", u_conv4_bias.data(), model::conv4_filters);

  // Load conv5 parameters
  readDataFromFile("sparse/features_10_weight.txt",
                   u_conv5_weights.data(),
                   model::conv5_filters * model::conv4_filters *
                       model::kernel_size * model::kernel_size);
  readDataFromFile(
      "sparse/features_10_bias.txt", u_conv5_bias.data(), model::conv5_filters);

  // Load linear layer parameters
  readDataFromFile("sparse/classifier_weight.txt",
                   u_linear_weights.data(),
                   model::num_classes * dims::flattened_size);
  readDataFromFile(
      "sparse/classifier_bias.txt", u_linear_bias.data(), model::num_classes);
}
