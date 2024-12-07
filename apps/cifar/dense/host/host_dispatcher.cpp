#include "host_dispatcher.hpp"

#include <spdlog/spdlog.h>

#include "apps/cifar/dense/app_data.hpp"
#include "host_kernels.hpp"

namespace cpu {

[[nodiscard]] core::multi_future<void> run_stage1(AppData &app_data,
                                                  core::thread_pool &pool,
                                                  const size_t n_threads) {
  spdlog::debug("CPU kernel 'conv2d', n = {}, threads = {}",
                app_data.u_image.size(),
                n_threads);

  // conv2d out_channels * out_height * out_width
  const auto output_channels = model::conv1_filters;

  return pool.submit_blocks(
      static_cast<size_t>(0),
      static_cast<size_t>(output_channels),
      [&](const size_t start, const size_t end) {
        // dense::conv2d_mt(data->u_image_data->data(),
        //                  data->conv1_dims.in_channels,
        //                  data->conv1_dims.in_height,
        //                  data->conv1_dims.in_width,
        //                  data->u_features_0_weights->data(),
        //                  data->conv1_dims.out_channels,
        //                  data->conv1_dims.in_channels,
        //                  kernel_size,
        //                  kernel_size,
        //                  data->u_features_0_biases->data(),
        //                  data->conv1_dims.out_channels,
        //                  kernel_size,
        //                  stride,
        //                  padding,
        //                  relu,
        //                  data->u_conv1_output->data(),
        //                  start,
        //                  end);

        // const float *input_data
        // const int image_input_channels
        // int input_height
        // int input_width
        // const float * weight_data
        // int weight_output_channels
        // int weight_input_channels
        // int weight_height
        // int weight_width
        // const float * bias_data
        // int bias_number_of_elements
        // int kernel_size
        // int stride
        // int padding
        // bool relu
        // float * output_data
        // int start
        // int end
        cpu::kernels::dense::conv2d_mt(
            app_data.u_image.data(),
            model::input_channels,
            model::input_height,
            model::input_width,
            app_data.u_conv1_weights.data(),
            output_channels,
            model::input_channels,
            model::kernel_size,
            model::kernel_size,
            app_data.u_conv1_bias.data(),
            model::kernel_size * model::kernel_size * model::input_channels,
            model::kernel_size,
            model::stride,
            model::padding,
            model::use_relu,
            app_data.u_conv1_output.data(),
            start,
            end);
      },
      n_threads);
}

}  // namespace cpu