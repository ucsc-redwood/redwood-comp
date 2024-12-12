#include "../../app.hpp"
#include "../arg_max.hpp"
#include "csr.hpp"
#include "sparse_app_data.hpp"
#include "tmp/original_kernels.hpp"

void run_cpu_demo() {
  auto mr = std::pmr::new_delete_resource();
  SparseAppData app_data(mr);

  static constexpr float sparsity_threshold = 1e-6;

  // Convert image to CSR
  const v1::CSRMatrix sparse_image =
      v1::denseToCsr(app_data.u_image.data(),
                     model::kInputChannels,
                     model::kInputHeight * model::kInputWidth,
                     sparsity_threshold);

  // Convert Conv1 weights to CSR
  const v1::CSRMatrix sparse_conv1_weights = v1::denseToCsr(
      app_data.u_conv1_weights.data(),
      model::kConv1OutChannels,
      model::kInputChannels * model::kKernelSize * model::kKernelSize,
      sparsity_threshold);

  // Conv1
  cpu::sparse::kernels::sparseConv2d(sparse_image,
                                     model::kInputHeight,
                                     model::kInputWidth,
                                     sparse_conv1_weights,
                                     model::kConv1OutChannels,
                                     app_data.u_conv1_bias.data(),
                                     model::kKernelSize,
                                     model::kStride,
                                     model::kPadding,
                                     model::kRelu,
                                     app_data.u_conv1_out.data());

  // Convert Conv1 output to CSR
  v1::CSRMatrix sparse_conv1_out =
      v1::denseToCsr(app_data.u_conv1_out.data(),
                     model::kConv1OutChannels,
                     model::kConv1OutHeight * model::kConv1OutWidth,
                     sparsity_threshold);

  // Pool1
  cpu::sparse::kernels::sparseMaxPool2d(sparse_conv1_out,
                                        model::kConv1OutChannels,
                                        model::kConv1OutHeight,
                                        model::kConv1OutWidth,
                                        model::kPoolSize,
                                        model::kPoolStride,
                                        app_data.u_pool1_out.data());

  v1::CSRMatrix sparse_pool1_out =
      v1::denseToCsr(app_data.u_pool1_out.data(),
                     model::kConv1OutChannels,
                     model::kPool1OutHeight * model::kPool1OutWidth,
                     sparsity_threshold);

  // Conv2
  cpu::sparse::kernels::sparseConv2d(sparse_pool1_out,
                                     model::kPool1OutHeight,
                                     model::kPool1OutWidth,
                                     app_data.sparse_conv2_weights,
                                     model::kConv2OutChannels,
                                     app_data.u_conv2_bias.data(),
                                     model::kKernelSize,
                                     model::kStride,
                                     model::kPadding,
                                     model::kRelu,
                                     app_data.u_conv2_out.data());

  v1::CSRMatrix sparse_conv2_out =
      v1::denseToCsr(app_data.u_conv2_out.data(),
                     model::kConv2OutChannels,
                     model::kConv2OutHeight * model::kConv2OutWidth,
                     sparsity_threshold);

  // Pool2
  cpu::sparse::kernels::sparseMaxPool2d(sparse_conv2_out,
                                        model::kConv2OutChannels,
                                        model::kConv2OutHeight,
                                        model::kConv2OutWidth,
                                        model::kPoolSize,
                                        model::kPoolStride,
                                        app_data.u_pool2_out.data());

  v1::CSRMatrix sparse_pool2_out =
      v1::denseToCsr(app_data.u_pool2_out.data(),
                     model::kConv2OutChannels,
                     model::kPool2OutHeight * model::kPool2OutWidth,
                     sparsity_threshold);

  // Conv3
  cpu::sparse::kernels::sparseConv2d(sparse_pool2_out,
                                     model::kPool2OutHeight,
                                     model::kPool2OutWidth,
                                     app_data.sparse_conv3_weights,
                                     model::kConv3OutChannels,
                                     app_data.u_conv3_bias.data(),
                                     model::kKernelSize,
                                     model::kStride,
                                     model::kPadding,
                                     model::kRelu,
                                     app_data.u_conv3_out.data());

  v1::CSRMatrix sparse_conv3_out =
      v1::denseToCsr(app_data.u_conv3_out.data(),
                     model::kConv3OutChannels,
                     model::kConv3OutHeight * model::kConv3OutWidth,
                     sparsity_threshold);

  // Conv4
  cpu::sparse::kernels::sparseConv2d(sparse_conv3_out,
                                     model::kConv3OutHeight,
                                     model::kConv3OutWidth,
                                     app_data.sparse_conv4_weights,
                                     model::kConv4OutChannels,
                                     app_data.u_conv4_bias.data(),
                                     model::kKernelSize,
                                     model::kStride,
                                     model::kPadding,
                                     model::kRelu,
                                     app_data.u_conv4_out.data());

  v1::CSRMatrix sparse_conv4_out =
      v1::denseToCsr(app_data.u_conv4_out.data(),
                     model::kConv4OutChannels,
                     model::kConv4OutHeight * model::kConv4OutWidth,
                     sparsity_threshold);

  // Conv5
  cpu::sparse::kernels::sparseConv2d(sparse_conv4_out,
                                     model::kConv4OutHeight,
                                     model::kConv4OutWidth,
                                     app_data.sparse_conv5_weights,
                                     model::kConv5OutChannels,
                                     app_data.u_conv5_bias.data(),
                                     model::kKernelSize,
                                     model::kStride,
                                     model::kPadding,
                                     model::kRelu,
                                     app_data.u_conv5_out.data());

  v1::CSRMatrix sparse_conv5_out =
      v1::denseToCsr(app_data.u_conv5_out.data(),
                     model::kConv5OutChannels,
                     model::kConv5OutHeight * model::kConv5OutWidth,
                     sparsity_threshold);

  // Pool3
  cpu::sparse::kernels::sparseMaxPool2d(sparse_conv5_out,
                                        model::kConv5OutChannels,
                                        model::kConv5OutHeight,
                                        model::kConv5OutWidth,
                                        model::kPoolSize,
                                        model::kPoolStride,
                                        app_data.u_pool3_out.data());

  // Flatten output for linear layer
  v1::CSRMatrix sparse_flattened = v1::denseToCsr(app_data.u_pool3_out.data(),
                                                  1,
                                                  model::kLinearInFeatures,
                                                  sparsity_threshold);

  // Linear
  cpu::sparse::kernels::sparseLinearLayer(sparse_flattened,
                                          app_data.sparse_linear_weights,
                                          app_data.u_linear_bias.data(),
                                          app_data.u_linear_out.data(),
                                          model::kLinearOutFeatures);

  print_prediction(arg_max(app_data.u_linear_out.data()));
}

int main(int argc, char** argv) {
  INIT_APP("cifar-sparse");

  run_cpu_demo();

  return 0;
}
