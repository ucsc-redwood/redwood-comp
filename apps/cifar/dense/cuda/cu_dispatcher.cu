#include <spdlog/spdlog.h>

#include "cu_dispatcher.cuh"
#include "cu_kernels.cuh"
#include "redwood/cuda/helpers.cuh"

namespace cuda {

// -----------------------------------------------------------------------------
// Stage 1 (first conv2d)
// -----------------------------------------------------------------------------

void run_stage1(AppData &app_data, const cudaStream_t stream) {
  const int total_iterations =
      model::kConv1OutChannels * model::kConv1OutHeight * model::kConv1OutWidth;

  static constexpr auto block_size = 256;
  const auto grid_size = div_up(total_iterations, block_size);
  constexpr auto s_mem = 0;

  spdlog::debug(
      "CUDA kernel 'conv2d_mt', n = {}, threads = {}, blocks = {}, stream: {}",
      total_iterations,
      block_size,
      grid_size,
      reinterpret_cast<void *>(stream));

  kernels::dense::conv2d<<<grid_size, block_size, s_mem, stream>>>(
      app_data.u_image.data(),
      app_data.u_conv1_weights.data(),
      app_data.u_conv1_bias.data(),
      app_data.u_conv1_out.data(),
      model::kInputHeight,
      model::kInputWidth,
      model::kConv1OutChannels,
      model::kInputChannels,
      model::kKernelSize,
      model::kKernelSize,
      model::kConv1BiasSize,
      model::kKernelSize,
      model::kStride,
      model::kPadding,
      model::kConv1OutHeight,
      model::kConv1OutWidth,
      model::kRelu);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// -----------------------------------------------------------------------------
// Stage 2 (maxpool)
// -----------------------------------------------------------------------------

void run_stage2(AppData &app_data, const cudaStream_t stream) {
  const int total_iterations =
      model::kConv1OutChannels * model::kPool1OutHeight * model::kPool1OutWidth;

  static constexpr auto block_size = 256;
  const auto grid_size = div_up(total_iterations, block_size);
  constexpr auto s_mem = 0;

  spdlog::debug(
      "CUDA kernel 'maxpool2d', n = {}, threads = {}, blocks = {}, stream: {}",
      total_iterations,
      block_size,
      grid_size,
      reinterpret_cast<void *>(stream));

  kernels::dense::maxpool2d<<<grid_size, block_size, s_mem, stream>>>(
      app_data.u_conv1_out.data(),
      app_data.u_pool1_out.data(),
      model::kConv1OutChannels,
      model::kConv1OutHeight,
      model::kConv1OutWidth,
      model::kPoolSize,
      model::kStride,
      model::kPool1OutHeight,
      model::kPool1OutWidth);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// -----------------------------------------------------------------------------
// Stage 3 (second conv2d)
// -----------------------------------------------------------------------------

void run_stage3(AppData &app_data, const cudaStream_t stream) {
  const int total_iterations =
      model::kConv2OutChannels * model::kConv2OutHeight * model::kConv2OutWidth;

  static constexpr auto block_size = 256;
  const auto grid_size = div_up(total_iterations, block_size);
  constexpr auto s_mem = 0;

  spdlog::debug(
      "CUDA kernel 'conv2d_mt', n = {}, threads = {}, blocks = {}, stream: {}",
      total_iterations,
      block_size,
      grid_size,
      reinterpret_cast<void *>(stream));

  kernels::dense::conv2d<<<grid_size, block_size, s_mem, stream>>>(
      app_data.u_pool1_out.data(),
      app_data.u_conv2_weights.data(),
      app_data.u_conv2_bias.data(),
      app_data.u_conv2_out.data(),
      model::kPool1OutHeight,
      model::kPool1OutWidth,
      model::kConv2OutChannels,
      model::kConv1OutChannels,
      model::kKernelSize,
      model::kKernelSize,
      model::kConv2BiasSize,
      model::kKernelSize,
      model::kStride,
      model::kPadding,
      model::kConv2OutHeight,
      model::kConv2OutWidth,
      model::kRelu);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// -----------------------------------------------------------------------------
// Stage 4 (second maxpool2d)
// -----------------------------------------------------------------------------

void run_stage4(AppData &app_data, const cudaStream_t stream) {
  const int total_iterations =
      model::kConv2OutChannels * model::kPool2OutHeight * model::kPool2OutWidth;

  static constexpr auto block_size = 256;
  const auto grid_size = div_up(total_iterations, block_size);
  constexpr auto s_mem = 0;

  spdlog::debug(
      "CUDA kernel 'maxpool2d', n = {}, threads = {}, blocks = {}, stream: {}",
      total_iterations,
      block_size,
      grid_size,
      reinterpret_cast<void *>(stream));

  kernels::dense::maxpool2d<<<grid_size, block_size, s_mem, stream>>>(
      app_data.u_conv2_out.data(),
      app_data.u_pool2_out.data(),
      model::kConv2OutChannels,
      model::kConv2OutHeight,
      model::kConv2OutWidth,
      model::kPoolSize,
      model::kStride,
      model::kPool2OutHeight,
      model::kPool2OutWidth);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// -----------------------------------------------------------------------------
// Stage 5 (third conv2d)
// -----------------------------------------------------------------------------

void run_stage5(AppData &app_data, const cudaStream_t stream) {
  const int total_iterations =
      model::kConv3OutChannels * model::kConv3OutHeight * model::kConv3OutWidth;

  static constexpr auto block_size = 256;
  const auto grid_size = div_up(total_iterations, block_size);
  constexpr auto s_mem = 0;

  spdlog::debug(
      "CUDA kernel 'conv2d_mt', n = {}, threads = {}, blocks = {}, stream: {}",
      total_iterations,
      block_size,
      grid_size,
      reinterpret_cast<void *>(stream));

  kernels::dense::conv2d<<<grid_size, block_size, s_mem, stream>>>(
      app_data.u_pool2_out.data(),
      app_data.u_conv3_weights.data(),
      app_data.u_conv3_bias.data(),
      app_data.u_conv3_out.data(),
      model::kPool2OutHeight,
      model::kPool2OutWidth,
      model::kConv3OutChannels,
      model::kConv2OutChannels,
      model::kKernelSize,
      model::kKernelSize,
      model::kConv3BiasSize,
      model::kKernelSize,
      model::kStride,
      model::kPadding,
      model::kConv3OutHeight,
      model::kConv3OutWidth,
      model::kRelu);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// -----------------------------------------------------------------------------
// Stage 6 (fourth conv2d)
// -----------------------------------------------------------------------------

void run_stage6(AppData &app_data, const cudaStream_t stream) {
  const int total_iterations =
      model::kConv4OutChannels * model::kConv4OutHeight * model::kConv4OutWidth;

  static constexpr auto block_size = 256;
  const auto grid_size = div_up(total_iterations, block_size);
  constexpr auto s_mem = 0;

  spdlog::debug(
      "CUDA kernel 'conv2d_mt', n = {}, threads = {}, blocks = {}, stream: {}",
      total_iterations,
      block_size,
      grid_size,
      reinterpret_cast<void *>(stream));

  kernels::dense::conv2d<<<grid_size, block_size, s_mem, stream>>>(
      app_data.u_conv3_out.data(),
      app_data.u_conv4_weights.data(),
      app_data.u_conv4_bias.data(),
      app_data.u_conv4_out.data(),
      model::kConv3OutHeight,
      model::kConv3OutWidth,
      model::kConv4OutChannels,
      model::kConv3OutChannels,
      model::kKernelSize,
      model::kKernelSize,
      model::kConv4BiasSize,
      model::kKernelSize,
      model::kStride,
      model::kPadding,
      model::kConv4OutHeight,
      model::kConv4OutWidth,
      model::kRelu);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// -----------------------------------------------------------------------------
// Stage 7 (fifth conv2d)
// -----------------------------------------------------------------------------

void run_stage7(AppData &app_data, const cudaStream_t stream) {
  const int total_iterations =
      model::kConv5OutChannels * model::kConv5OutHeight * model::kConv5OutWidth;

  static constexpr auto block_size = 256;
  const auto grid_size = div_up(total_iterations, block_size);
  constexpr auto s_mem = 0;

  spdlog::debug(
      "CUDA kernel 'conv2d_mt', n = {}, threads = {}, blocks = {}, stream: {}",
      total_iterations,
      block_size,
      grid_size,
      reinterpret_cast<void *>(stream));

  kernels::dense::conv2d<<<grid_size, block_size, s_mem, stream>>>(
      app_data.u_conv4_out.data(),
      app_data.u_conv5_weights.data(),
      app_data.u_conv5_bias.data(),
      app_data.u_conv5_out.data(),
      model::kConv4OutHeight,
      model::kConv4OutWidth,
      model::kConv5OutChannels,
      model::kConv4OutChannels,
      model::kKernelSize,
      model::kKernelSize,
      model::kConv5BiasSize,
      model::kKernelSize,
      model::kStride,
      model::kPadding,
      model::kConv5OutHeight,
      model::kConv5OutWidth,
      model::kRelu);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// -----------------------------------------------------------------------------
// Stage 8 (third maxpool2d)
// -----------------------------------------------------------------------------

void run_stage8(AppData &app_data, const cudaStream_t stream) {
  const int total_iterations =
      model::kConv5OutChannels * model::kPool3OutHeight * model::kPool3OutWidth;

  static constexpr auto block_size = 256;
  const auto grid_size = div_up(total_iterations, block_size);
  constexpr auto s_mem = 0;

  spdlog::debug(
      "CUDA kernel 'maxpool2d', n = {}, threads = {}, blocks = {}, stream: {}",
      total_iterations,
      block_size,
      grid_size,
      reinterpret_cast<void *>(stream));

  kernels::dense::maxpool2d<<<grid_size, block_size, s_mem, stream>>>(
      app_data.u_conv5_out.data(),
      app_data.u_pool3_out.data(),
      model::kConv5OutChannels,
      model::kConv5OutHeight,
      model::kConv5OutWidth,
      model::kPoolSize,
      model::kStride,
      model::kPool3OutHeight,
      model::kPool3OutWidth);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// -----------------------------------------------------------------------------
// Stage 9 (linear)
// -----------------------------------------------------------------------------

void run_stage9(AppData &app_data, const cudaStream_t stream) {
  const int total_iterations = model::kLinearOutFeatures;

  static constexpr auto block_size = 256;
  const auto grid_size = div_up(total_iterations, block_size);
  constexpr auto s_mem = 0;

  spdlog::debug(
      "CUDA kernel 'linear', n = {}, threads = {}, blocks = {}, stream: {}",
      total_iterations,
      block_size,
      grid_size,
      reinterpret_cast<void *>(stream));

  kernels::dense::linear<<<grid_size, block_size, s_mem, stream>>>(
      app_data.u_pool3_out.data(),
      app_data.u_linear_weights.data(),
      app_data.u_linear_bias.data(),
      app_data.u_linear_out.data(),
      model::kLinearInFeatures,
      model::kLinearOutFeatures);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace cuda
