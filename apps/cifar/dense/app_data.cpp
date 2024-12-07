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

using namespace model;

AppData::AppData(std::pmr::memory_resource* mr)
    : BaseAppData(mr),
      u_image(kImageSize, mr),
      u_conv1_weights(kConv1WeightSize, mr),
      u_conv1_bias(kConv1BiasSize, mr),
      u_conv1_out(kConv1OutSize, mr),
      u_pool1_out(kPool1OutSize, mr),
      u_conv2_weights(kConv2WeightSize, mr),
      u_conv2_bias(kConv2BiasSize, mr),
      u_conv2_out(kConv2OutSize, mr),
      u_pool2_out(kPool2OutSize, mr),
      u_conv3_weights(kConv3WeightSize, mr),
      u_conv3_bias(kConv3BiasSize, mr),
      u_conv3_out(kConv3OutSize, mr),
      u_conv4_weights(kConv4WeightSize, mr),
      u_conv4_bias(kConv4BiasSize, mr),
      u_conv4_out(kConv4OutSize, mr),
      u_conv5_weights(kConv5WeightSize, mr),
      u_conv5_bias(kConv5BiasSize, mr),
      u_conv5_out(kConv5OutSize, mr),
      u_pool3_out(kPool3OutSize, mr),
      u_linear_weights(kLinearWeightSize, mr),
      u_linear_bias(kLinearBiasSize, mr),
      u_linear_out(kLinearOutSize, mr) {
  // Load input image
  readDataFromFile(
      "images/flattened_bird_bird_57.txt", u_image.data(), kImageSize);

  // Load conv1 parameters
  readDataFromFile(
      "sparse/features_0_weight.txt", u_conv1_weights.data(), kConv1WeightSize);
  readDataFromFile(
      "sparse/features_0_bias.txt", u_conv1_bias.data(), kConv1BiasSize);

  // Load conv2 parameters
  readDataFromFile(
      "sparse/features_3_weight.txt", u_conv2_weights.data(), kConv2WeightSize);
  readDataFromFile(
      "sparse/features_3_bias.txt", u_conv2_bias.data(), kConv2BiasSize);

  // Load conv3 parameters
  readDataFromFile(
      "sparse/features_6_weight.txt", u_conv3_weights.data(), kConv3WeightSize);
  readDataFromFile(
      "sparse/features_6_bias.txt", u_conv3_bias.data(), kConv3BiasSize);

  // Load conv4 parameters
  readDataFromFile(
      "sparse/features_8_weight.txt", u_conv4_weights.data(), kConv4WeightSize);
  readDataFromFile(
      "sparse/features_8_bias.txt", u_conv4_bias.data(), kConv4BiasSize);

  // Load conv5 parameters
  readDataFromFile("sparse/features_10_weight.txt",
                   u_conv5_weights.data(),
                   kConv5WeightSize);
  readDataFromFile(
      "sparse/features_10_bias.txt", u_conv5_bias.data(), kConv5BiasSize);

  // Load linear parameters
  readDataFromFile("sparse/classifier_weight.txt",
                   u_linear_weights.data(),
                   kLinearWeightSize);
  readDataFromFile(
      "sparse/classifier_bias.txt", u_linear_bias.data(), kLinearBiasSize);
}
