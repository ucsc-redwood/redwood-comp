#include "host_dispatcher.hpp"

#include <spdlog/spdlog.h>

#include "host_kernels.hpp"

namespace cpu {

// ----------------------------------------------------------------------------
// Stage 1 (first conv2d)
// ----------------------------------------------------------------------------
// We know from model:
// Input: buffers.image (3 x 32 x 32)
// Weights: buffers.conv1_weights (64 out_channels, 3 in_channels, 3x3 kernel)
// Bias: buffers.conv1_bias (64)
// Output: buffers.conv1_out (64 x 32 x 32)
// ----------------------------------------------------------------------------

auto run_stage1(AppData &app_data,
                core::thread_pool &pool,
                const size_t n_threads) -> core::multi_future<void> {
  const int total_iterations =
      model::kConv1OutChannels * model::kConv1OutHeight * model::kConv1OutWidth;

  const int start = 0;
  const int end = total_iterations;

  spdlog::debug(
      "Conv2D: ({} x {} x {}) ---> ({} x {} x {}), Threads: {}, Total "
      "iterations: {}",
      model::kInputChannels,
      model::kInputHeight,
      model::kInputWidth,
      model::kConv1OutChannels,
      model::kConv1OutHeight,
      model::kConv1OutWidth,
      n_threads,
      total_iterations);

  return pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        kernels::dense::conv2d_mt(
            app_data.u_image.data(),
            model::kInputChannels,  // image_input_channels
            model::kInputHeight,
            model::kInputWidth,
            app_data.u_conv1_weights.data(),
            model::kConv1OutChannels,
            model::kInputChannels,
            model::kKernelSize,  // weight_height
            model::kKernelSize,  // weight_width
            app_data.u_conv1_bias.data(),
            model::kConv1BiasSize,
            model::kKernelSize,
            model::kStride,
            model::kPadding,
            model::kRelu,
            app_data.u_conv1_out.data(),
            start,
            end);
      },
      n_threads);
}

// ----------------------------------------------------------------------------
// Stage 2 (first maxpool2d)
// ----------------------------------------------------------------------------
// We know from model:
// Input: buffers.conv1_out (64 x 32 x 32)
// Output: buffers.pool1_out (64 x 16 x 16)
// ----------------------------------------------------------------------------

auto run_stage2(AppData &app_data,
                core::thread_pool &pool,
                const size_t n_threads) -> core::multi_future<void> {
  const int total_iterations =
      model::kConv1OutChannels * model::kPool1OutHeight * model::kPool1OutWidth;

  const int start = 0;
  const int end = total_iterations;

  spdlog::debug(
      "MaxPool2D: ({} x {} x {}) ---> ({} x {} x {}), Threads: {}, Total "
      "iterations: {}",
      model::kConv1OutChannels,
      model::kPool1OutHeight,
      model::kPool1OutWidth,
      model::kConv1OutChannels,
      model::kPool1OutHeight,
      model::kPool1OutWidth,
      n_threads,
      total_iterations);

  return pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        kernels::dense::maxpool2d_mt(app_data.u_conv1_out.data(),
                                     model::kConv1OutChannels,
                                     model::kConv1OutHeight,
                                     model::kConv1OutWidth,
                                     model::kPoolSize,
                                     model::kPoolStride,
                                     app_data.u_pool1_out.data(),
                                     start,
                                     end);
      },
      n_threads);
}

// ----------------------------------------------------------------------------
// Stage 3 (second conv2d)
// ----------------------------------------------------------------------------
// We know from model:
// Input: buffers.pool1_out (64 x 16 x 16)
// Weights: buffers.conv2_weights (128 out_channels, 64 in_channels, 3x3 kernel)
// Bias: buffers.conv2_bias (128)
// Output: buffers.conv2_out (128 x 16 x 16)
// ----------------------------------------------------------------------------

auto run_stage3(AppData &app_data,
                core::thread_pool &pool,
                const size_t n_threads) -> core::multi_future<void> {
  const int total_iterations =
      model::kConv2OutChannels * model::kConv2OutHeight * model::kConv2OutWidth;

  const int start = 0;
  const int end = total_iterations;

  spdlog::debug(
      "Conv2D: ({} x {} x {}) ---> ({} x {} x {}), Threads: {}, Total "
      "iterations: {}",
      model::kConv1OutChannels,
      model::kPool1OutHeight,
      model::kPool1OutWidth,
      model::kConv2OutChannels,
      model::kConv2OutHeight,
      model::kConv2OutWidth,
      n_threads,
      total_iterations);

  return pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        kernels::dense::conv2d_mt(app_data.u_pool1_out.data(),
                                  model::kConv1OutChannels,
                                  model::kPool1OutHeight,
                                  model::kPool1OutWidth,
                                  app_data.u_conv2_weights.data(),
                                  model::kConv2OutChannels,
                                  model::kConv1OutChannels,
                                  model::kKernelSize,  // weight_height
                                  model::kKernelSize,  // weight_width
                                  app_data.u_conv2_bias.data(),
                                  model::kConv2BiasSize,
                                  model::kKernelSize,
                                  model::kStride,
                                  model::kPadding,
                                  model::kRelu,
                                  app_data.u_conv2_out.data(),
                                  start,
                                  end);
      },
      n_threads);
}

// ----------------------------------------------------------------------------
// Stage 4 (second maxpool2d)
// ----------------------------------------------------------------------------

auto run_stage4(AppData &app_data,
                core::thread_pool &pool,
                const size_t n_threads) -> core::multi_future<void> {
  const int total_iterations =
      model::kConv2OutChannels * model::kPool2OutHeight * model::kPool2OutWidth;

  const int start = 0;
  const int end = total_iterations;

  spdlog::debug(
      "MaxPool2D: ({} x {} x {}) ---> ({} x {} x {}), Threads: {}, Total "
      "iterations: {}",
      model::kConv2OutChannels,
      model::kConv2OutHeight,
      model::kConv2OutWidth,
      model::kConv2OutChannels,
      model::kPool2OutHeight,
      model::kPool2OutWidth,
      n_threads,
      total_iterations);

  return pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        kernels::dense::maxpool2d_mt(app_data.u_conv2_out.data(),
                                     model::kConv2OutChannels,
                                     model::kConv2OutHeight,
                                     model::kConv2OutWidth,
                                     model::kPoolSize,
                                     model::kPoolStride,
                                     app_data.u_pool2_out.data(),
                                     start,
                                     end);
      },
      n_threads);
}

// ----------------------------------------------------------------------------
// Stage 5 (third conv2d)
// ----------------------------------------------------------------------------

auto run_stage5(AppData &app_data,
                core::thread_pool &pool,
                const size_t n_threads) -> core::multi_future<void> {
  const int total_iterations =
      model::kConv3OutChannels * model::kConv3OutHeight * model::kConv3OutWidth;

  const int start = 0;
  const int end = total_iterations;

  spdlog::debug(
      "Conv2D: ({} x {} x {}) ---> ({} x {} x {}), Threads: {}, Total "
      "iterations: {}",
      model::kConv2OutChannels,
      model::kPool2OutHeight,
      model::kPool2OutWidth,
      model::kConv3OutChannels,
      model::kConv3OutHeight,
      model::kConv3OutWidth,
      n_threads,
      total_iterations);

  return pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        kernels::dense::conv2d_mt(app_data.u_pool2_out.data(),
                                  model::kConv2OutChannels,
                                  model::kPool2OutHeight,
                                  model::kPool2OutWidth,
                                  app_data.u_conv3_weights.data(),
                                  model::kConv3OutChannels,
                                  model::kConv2OutChannels,
                                  model::kKernelSize,
                                  model::kKernelSize,
                                  app_data.u_conv3_bias.data(),
                                  model::kConv3BiasSize,
                                  model::kKernelSize,
                                  model::kStride,
                                  model::kPadding,
                                  model::kRelu,
                                  app_data.u_conv3_out.data(),
                                  start,
                                  end);
      },
      n_threads);
}

// ----------------------------------------------------------------------------
// Stage 6 (fourth conv2d)
// ----------------------------------------------------------------------------

auto run_stage6(AppData &app_data,
                core::thread_pool &pool,
                const size_t n_threads) -> core::multi_future<void> {
  const int total_iterations =
      model::kConv4OutChannels * model::kConv4OutHeight * model::kConv4OutWidth;

  const int start = 0;
  const int end = total_iterations;

  spdlog::debug(
      "Conv2D: ({} x {} x {}) ---> ({} x {} x {}), Threads: {}, Total "
      "iterations: {}",
      model::kConv3OutChannels,
      model::kConv3OutHeight,
      model::kConv3OutWidth,
      model::kConv4OutChannels,
      model::kConv4OutHeight,
      model::kConv4OutWidth,
      n_threads,
      total_iterations);

  return pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        kernels::dense::conv2d_mt(app_data.u_conv3_out.data(),
                                  model::kConv3OutChannels,
                                  model::kConv3OutHeight,
                                  model::kConv3OutWidth,
                                  app_data.u_conv4_weights.data(),
                                  model::kConv4OutChannels,
                                  model::kConv3OutChannels,
                                  model::kKernelSize,
                                  model::kKernelSize,
                                  app_data.u_conv4_bias.data(),
                                  model::kConv4BiasSize,
                                  model::kKernelSize,
                                  model::kStride,
                                  model::kPadding,
                                  model::kRelu,
                                  app_data.u_conv4_out.data(),
                                  start,
                                  end);
      },
      n_threads);
}

// ----------------------------------------------------------------------------
// Stage 7 (fifth conv2d)
// ----------------------------------------------------------------------------

auto run_stage7(AppData &app_data,
                core::thread_pool &pool,
                const size_t n_threads) -> core::multi_future<void> {
  const int total_iterations =
      model::kConv5OutChannels * model::kConv5OutHeight * model::kConv5OutWidth;

  const int start = 0;
  const int end = total_iterations;

  spdlog::debug(
      "Conv2D: ({} x {} x {}) ---> ({} x {} x {}), Threads: {}, Total "
      "iterations: {}",
      model::kConv4OutChannels,
      model::kConv4OutHeight,
      model::kConv4OutWidth,
      model::kConv5OutChannels,
      model::kConv5OutHeight,
      model::kConv5OutWidth,
      n_threads,
      total_iterations);

  return pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        kernels::dense::conv2d_mt(app_data.u_conv4_out.data(),
                                  model::kConv4OutChannels,
                                  model::kConv4OutHeight,
                                  model::kConv4OutWidth,
                                  app_data.u_conv5_weights.data(),
                                  model::kConv5OutChannels,
                                  model::kConv4OutChannels,
                                  model::kKernelSize,
                                  model::kKernelSize,
                                  app_data.u_conv5_bias.data(),
                                  model::kConv5BiasSize,
                                  model::kKernelSize,
                                  model::kStride,
                                  model::kPadding,
                                  model::kRelu,
                                  app_data.u_conv5_out.data(),
                                  start,
                                  end);
      },
      n_threads);
}

// ----------------------------------------------------------------------------
// Stage 8 (third maxpool2d)
// ----------------------------------------------------------------------------

auto run_stage8(AppData &app_data,
                core::thread_pool &pool,
                const size_t n_threads) -> core::multi_future<void> {
  const int total_iterations =
      model::kConv5OutChannels * model::kPool3OutHeight * model::kPool3OutWidth;

  const int start = 0;
  const int end = total_iterations;

  spdlog::debug(
      "MaxPool2D: ({} x {} x {}) ---> ({} x {} x {}), Threads: {}, Total "
      "iterations: {}",
      model::kConv5OutChannels,
      model::kConv5OutHeight,
      model::kConv5OutWidth,
      model::kConv5OutChannels,
      model::kPool3OutHeight,
      model::kPool3OutWidth,
      n_threads,
      total_iterations);

  return pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        kernels::dense::maxpool2d_mt(app_data.u_conv5_out.data(),
                                     model::kConv5OutChannels,
                                     model::kConv5OutHeight,
                                     model::kConv5OutWidth,
                                     model::kPoolSize,
                                     model::kPoolStride,
                                     app_data.u_pool3_out.data(),
                                     start,
                                     end);
      },
      n_threads);
}

// ----------------------------------------------------------------------------
// Stage 9 (linear)
// ----------------------------------------------------------------------------

auto run_stage9(AppData &app_data,
                core::thread_pool &pool,
                const size_t n_threads) -> core::multi_future<void> {
  const int total_iterations = model::kLinearOutFeatures;

  const int start = 0;
  const int end = total_iterations;

  spdlog::debug("Linear: ({}) ---> ({}), Threads: {}, Total iterations: {}",
                model::kLinearInFeatures,
                model::kLinearOutFeatures,
                n_threads,
                total_iterations);

  return pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        kernels::dense::linear_mt(app_data.u_pool3_out.data(),
                                  app_data.u_linear_weights.data(),
                                  app_data.u_linear_bias.data(),
                                  app_data.u_linear_out.data(),
                                  model::kLinearInFeatures,
                                  model::kLinearOutFeatures,
                                  start,
                                  end);
      },
      n_threads);
}

}  // namespace cpu