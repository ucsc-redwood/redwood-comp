#include "vk_dispatcher.hpp"

#include "apps/cifar/dense/app_data.hpp"

namespace vulkan {

Dispatcher::Dispatcher(Engine &engine, AppData &app_data)
    : engine_ref(engine), app_data_ref(app_data) {
  auto conv2d_algo =
      engine
          .algorithm("cifar_conv2d.comp",
                     {
                         // We still need the buffer here, because we need to
                         // know the size to setup the vk::Pipeline. While the
                         // values here does not matter yet.
                         engine.get_buffer(app_data.u_conv1_out.data()),
                         engine.get_buffer(app_data.u_conv2_weights.data()),
                         engine.get_buffer(app_data.u_conv2_bias.data()),
                         engine.get_buffer(app_data.u_conv2_out.data()),
                     })
          ->set_push_constants<Conv2dPushConstants>({
              // Similarly here, we need to know how many elements we have in
              .input_height = model::kInputHeight,
              .input_width = model::kInputWidth,
              .weight_output_channels = model::kConv1OutChannels,
              .weight_input_channels = model::kInputChannels,
              .weight_height = model::kKernelSize,
              .weight_width = model::kKernelSize,
              .bias_number_of_elements = model::kConv1BiasSize,
              .kernel_size = model::kKernelSize,
              .stride = model::kStride,
              .padding = model::kPadding,
              .output_height = model::kConv1OutHeight,
              .output_width = model::kConv1OutWidth,
              .relu = model::kRelu,
          })
          ->build();

  algorithms.try_emplace("conv2d", std::move(conv2d_algo));

  auto maxpool2d_algo =
      engine
          .algorithm(
              "cifar_maxpool2d.comp",
              {
                  engine.get_buffer(app_data.u_conv1_out.data()),  // input
                  engine.get_buffer(app_data.u_pool1_out.data()),  // output
              })
          ->set_push_constants<MaxpoolPushConstants>({
              .input_channels = model::kConv1OutChannels,
              .input_height = model::kConv1OutHeight,
              .input_width = model::kConv1OutWidth,
              .pool_size = model::kPoolSize,
              .stride = model::kPoolStride,
              .output_height = model::kPool1OutHeight,
              .output_width = model::kPool1OutWidth,
          })
          ->build();

  algorithms.try_emplace("maxpool2d", std::move(maxpool2d_algo));
}

// ----------------------------------------------------------------------------
// Stage 1 (first conv2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage1(Sequence *seq) {
  const int total_iterations =
      model::kConv1OutChannels * model::kConv1OutHeight * model::kConv1OutWidth;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_conv1_out.data()),
      engine_ref.get_buffer(app_data_ref.u_conv2_weights.data()),
      engine_ref.get_buffer(app_data_ref.u_conv2_bias.data()),
      engine_ref.get_buffer(app_data_ref.u_conv2_out.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
      .input_height = model::kInputHeight,
      .input_width = model::kInputWidth,
      .weight_output_channels = model::kConv1OutChannels,
      .weight_input_channels = model::kInputChannels,
      .weight_height = model::kKernelSize,
      .weight_width = model::kKernelSize,
      .bias_number_of_elements = model::kConv1BiasSize,
      .kernel_size = model::kKernelSize,
      .stride = model::kStride,
      .padding = model::kPadding,
      .output_height = model::kConv1OutHeight,
      .output_width = model::kConv1OutWidth,
      .relu = model::kRelu,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 2 (first maxpool2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage2(Sequence *seq) {
  const int total_iterations =
      model::kConv1OutChannels * model::kPool1OutHeight * model::kPool1OutWidth;

  auto algo = algorithms.at("maxpool2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_conv1_out.data()),
      engine_ref.get_buffer(app_data_ref.u_pool1_out.data()),
  });

  algo->update_push_constants(MaxpoolPushConstants{
      .input_channels = model::kConv1OutChannels,
      .input_height = model::kConv1OutHeight,
      .input_width = model::kConv1OutWidth,
      .pool_size = model::kPoolSize,
      .stride = model::kPoolStride,
      .output_height = model::kPool1OutHeight,
      .output_width = model::kPool1OutWidth,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 3 (second conv2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage3(Sequence *seq) {
  const int total_iterations =
      model::kConv2OutChannels * model::kConv2OutHeight * model::kConv2OutWidth;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_pool1_out.data()),
      engine_ref.get_buffer(app_data_ref.u_conv2_weights.data()),
      engine_ref.get_buffer(app_data_ref.u_conv2_bias.data()),
      engine_ref.get_buffer(app_data_ref.u_conv2_out.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
      .input_height = model::kPool1OutHeight,
      .input_width = model::kPool1OutWidth,
      .weight_output_channels = model::kConv2OutChannels,
      .weight_input_channels = model::kConv1OutChannels,
      .weight_height = model::kKernelSize,
      .weight_width = model::kKernelSize,
      .bias_number_of_elements = model::kConv2BiasSize,
      .kernel_size = model::kKernelSize,
      .stride = model::kStride,
      .padding = model::kPadding,
      .output_height = model::kConv2OutHeight,
      .output_width = model::kConv2OutWidth,
      .relu = model::kRelu,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();
  seq->sync();
}



}  // namespace vulkan
