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



}  // namespace cuda
