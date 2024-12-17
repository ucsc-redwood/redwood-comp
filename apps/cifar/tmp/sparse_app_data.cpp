#include "sparse_app_data.hpp"

SparseAppData::SparseAppData(std::pmr::memory_resource* mr) : AppData(mr) {
  sparse_image = v1::denseToCsr(u_image.data(),
                                model::kInputChannels,
                                model::kInputHeight * model::kInputWidth,
                                model::sparse_threshold);

  sparse_conv1_weights = v1::denseToCsr(
      u_conv1_weights.data(),
      model::kConv1OutChannels,
      model::kInputChannels * model::kKernelSize * model::kKernelSize,
      model::sparse_threshold);

  sparse_conv2_weights = v1::denseToCsr(
      u_conv2_weights.data(),
      model::kConv2OutChannels,
      model::kConv1OutChannels * model::kKernelSize * model::kKernelSize,
      model::sparse_threshold);

  sparse_conv3_weights = v1::denseToCsr(
      u_conv3_weights.data(),
      model::kConv3OutChannels,
      model::kConv2OutChannels * model::kKernelSize * model::kKernelSize,
      model::sparse_threshold);

  sparse_conv4_weights = v1::denseToCsr(
      u_conv4_weights.data(),
      model::kConv4OutChannels,
      model::kConv3OutChannels * model::kKernelSize * model::kKernelSize,
      model::sparse_threshold);

  sparse_conv5_weights = v1::denseToCsr(
      u_conv5_weights.data(),
      model::kConv5OutChannels,
      model::kConv4OutChannels * model::kKernelSize * model::kKernelSize,
      model::sparse_threshold);

  sparse_linear_weights = v1::denseToCsr(u_linear_weights.data(),
                                         model::kLinearOutFeatures,
                                         model::kLinearInFeatures,
                                         model::sparse_threshold);
}
