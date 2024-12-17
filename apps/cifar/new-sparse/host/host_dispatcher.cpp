#include "host_dispatcher.hpp"

#include <spdlog/spdlog.h>

#include "host_kernels.hpp"

void print_kernel_params(std::string_view name,
                         int start,
                         int end,
                         int n_threads) {
  spdlog::debug("{}: range [{}, {}), threads: {}, items/thread: ~{}",
                name,
                start,
                end,
                n_threads,
                (end - start) / n_threads);
}

namespace cpu {

namespace kernels {

namespace sparse {

void run_stage1(AppData& app_data,
                core::thread_pool& pool,
                const size_t n_threads,
                const bool sync) {
  constexpr auto start = 0;
  const auto end = app_data.conv1_weights.rows;

  print_kernel_params("stage1: conv2d_mt", start, end, n_threads);

  auto ret = pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        cpu::kernels::sparse::conv2d_mt(app_data.u_image_data.data(),
                                        3,
                                        32,
                                        32,
                                        app_data.conv1_weights,
                                        app_data.u_conv1_bias.data(),
                                        64,
                                        3,
                                        1,
                                        1,
                                        true,
                                        app_data.u_conv1_output.data(),
                                        start,
                                        end);
      },
      n_threads);

  if (sync) {
    ret.wait();
  }
}

void run_stage2(AppData& app_data,
                core::thread_pool& pool,
                size_t n_threads,
                const bool sync) {
  // start, end = 0, input_channels * output_height * output_width

  constexpr auto start = 0;
  constexpr auto end = 64 * 32 * 32;

  print_kernel_params("stage2: maxpool2d_mt", start, end, n_threads);

  auto ret = pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        cpu::kernels::sparse::maxpool2d_mt(app_data.u_conv1_output.data(),
                                           64,
                                           32,
                                           32,
                                           2,
                                           2,
                                           app_data.u_pool1_output.data(),
                                           start,
                                           end);
      },
      n_threads);

  if (sync) {
    ret.wait();
  }
}

void run_stage3(AppData& app_data,
                core::thread_pool& pool,
                size_t n_threads,
                const bool sync) {
  const auto start = 0;
  const auto end = app_data.conv2_weights.rows;

  print_kernel_params("stage3: conv2d_mt", start, end, n_threads);

  auto ret = pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        cpu::kernels::sparse::conv2d_mt(app_data.u_pool1_output.data(),
                                        64,
                                        16,
                                        16,
                                        app_data.conv2_weights,
                                        app_data.u_conv2_bias.data(),
                                        192,
                                        3,
                                        1,
                                        1,
                                        true,
                                        app_data.u_conv2_output.data(),
                                        start,
                                        end);
      },
      n_threads);

  if (sync) {
    ret.wait();
  }
}

void run_stage4(AppData& app_data,
                core::thread_pool& pool,
                size_t n_threads,
                const bool sync) {
  const auto start = 0;
  const auto end = 192 * 16 * 16;

  print_kernel_params("stage4: maxpool2d_mt", start, end, n_threads);

  auto ret = pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        cpu::kernels::sparse::maxpool2d_mt(app_data.u_conv2_output.data(),
                                           192,
                                           16,
                                           16,
                                           2,
                                           2,
                                           app_data.u_pool2_output.data(),
                                           start,
                                           end);
      },
      n_threads);

  if (sync) {
    ret.wait();
  }
}

void run_stage5(AppData& app_data,
                core::thread_pool& pool,
                size_t n_threads,
                const bool sync) {
  const auto start = 0;
  const auto end = app_data.conv3_weights.rows;

  print_kernel_params("stage5: conv2d_mt", start, end, n_threads);

  auto ret = pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        cpu::kernels::sparse::conv2d_mt(app_data.u_pool2_output.data(),
                                        192,
                                        8,
                                        8,
                                        app_data.conv3_weights,
                                        app_data.u_conv3_bias.data(),
                                        384,
                                        3,
                                        1,
                                        1,
                                        true,
                                        app_data.u_conv3_output.data(),
                                        start,
                                        end);
      },
      n_threads);

  if (sync) {
    ret.wait();
  }
}

void run_stage6(AppData& app_data,
                core::thread_pool& pool,
                size_t n_threads,
                const bool sync) {
  const auto start = 0;
  const auto end = app_data.conv4_weights.rows;

  print_kernel_params("stage6: conv2d_mt", start, end, n_threads);

  auto ret = pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        cpu::kernels::sparse::conv2d_mt(app_data.u_conv3_output.data(),
                                        384,
                                        8,
                                        8,
                                        app_data.conv4_weights,
                                        app_data.u_conv4_bias.data(),
                                        256,
                                        3,
                                        1,
                                        1,
                                        true,
                                        app_data.u_conv4_output.data(),
                                        start,
                                        end);
      },
      n_threads);

  if (sync) {
    ret.wait();
  }
}

void run_stage7(AppData& app_data,
                core::thread_pool& pool,
                size_t n_threads,
                const bool sync) {
  const auto start = 0;
  const auto end = app_data.conv5_weights.rows;

  print_kernel_params("stage7: conv2d_mt", start, end, n_threads);

  auto ret = pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        cpu::kernels::sparse::conv2d_mt(app_data.u_conv4_output.data(),
                                        256,
                                        8,
                                        8,
                                        app_data.conv5_weights,
                                        app_data.u_conv5_bias.data(),
                                        256,
                                        3,
                                        1,
                                        1,
                                        true,
                                        app_data.u_conv5_output.data(),
                                        start,
                                        end);
      },
      n_threads);

  if (sync) {
    ret.wait();
  }
}

void run_stage8(AppData& app_data,
                core::thread_pool& pool,
                size_t n_threads,
                const bool sync) {
  const auto start = 0;
  const auto end = 256 * 8 * 8;

  print_kernel_params("stage8: maxpool2d_mt", start, end, n_threads);

  auto ret = pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        cpu::kernels::sparse::maxpool2d_mt(app_data.u_conv5_output.data(),
                                           256,
                                           8,
                                           8,
                                           2,
                                           2,
                                           app_data.u_pool3_output.data(),
                                           start,
                                           end);
      },
      n_threads);

  if (sync) {
    ret.wait();
  }
}

void run_stage9(AppData& app_data,
                core::thread_pool& pool,
                size_t n_threads,
                const bool sync) {
  const auto start = 0;
  const auto end = app_data.linear_weights.rows;

  print_kernel_params("stage9: linear_mt", start, end, n_threads);

  auto ret = pool.submit_blocks(
      start,
      end,
      [&](const int start, const int end) {
        cpu::kernels::sparse::linear_mt(app_data.u_pool3_output.data(),
                                        app_data.linear_weights,
                                        app_data.u_linear_bias.data(),
                                        app_data.u_linear_output.data(),
                                        start,
                                        end);
      },
      n_threads);

  if (sync) {
    ret.wait();
  }
}

}  // namespace sparse
}  // namespace kernels
}  // namespace cpu
