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

  auto linear_algo =
      engine
          .algorithm(
              "cifar_linear.comp",
              {
                  engine.get_buffer(app_data.u_pool3_out.data()),  // input
                  engine.get_buffer(
                      app_data.u_linear_weights.data()),             // weights
                  engine.get_buffer(app_data.u_linear_bias.data()),  // bias
                  engine.get_buffer(app_data.u_linear_out.data()),   // output
              })
          ->set_push_constants<LinearPushConstants>({
              .in_features = model::kLinearInFeatures,
              .out_features = model::kLinearOutFeatures,
          })
          ->build();

  algorithms.try_emplace("linear", std::move(linear_algo));
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

// ----------------------------------------------------------------------------
// Stage 4 (second maxpool2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage4(Sequence *seq) {
  const int total_iterations =
      model::kConv2OutChannels * model::kPool2OutHeight * model::kPool2OutWidth;

  auto algo = algorithms.at("maxpool2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_conv2_out.data()),
      engine_ref.get_buffer(app_data_ref.u_pool2_out.data()),
  });

  algo->update_push_constants(MaxpoolPushConstants{
      .input_channels = model::kConv2OutChannels,
      .input_height = model::kConv2OutHeight,
      .input_width = model::kConv2OutWidth,
      .pool_size = model::kPoolSize,
      .stride = model::kPoolStride,
      .output_height = model::kPool2OutHeight,
      .output_width = model::kPool2OutWidth,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 5 (third conv2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage5(Sequence *seq) {
  const int total_iterations =
      model::kConv3OutChannels * model::kConv3OutHeight * model::kConv3OutWidth;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_pool2_out.data()),
      engine_ref.get_buffer(app_data_ref.u_conv3_weights.data()),
      engine_ref.get_buffer(app_data_ref.u_conv3_bias.data()),
      engine_ref.get_buffer(app_data_ref.u_conv3_out.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
      .input_height = model::kPool2OutHeight,
      .input_width = model::kPool2OutWidth,
      .weight_output_channels = model::kConv3OutChannels,
      .weight_input_channels = model::kConv2OutChannels,
      .weight_height = model::kKernelSize,
      .weight_width = model::kKernelSize,
      .bias_number_of_elements = model::kConv3BiasSize,
      .kernel_size = model::kKernelSize,
      .stride = model::kStride,
      .padding = model::kPadding,
      .output_height = model::kConv3OutHeight,
      .output_width = model::kConv3OutWidth,
      .relu = model::kRelu,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 6 (fourth conv2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage6(Sequence *seq) {
  const int total_iterations =
      model::kConv4OutChannels * model::kConv4OutHeight * model::kConv4OutWidth;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_conv3_out.data()),
      engine_ref.get_buffer(app_data_ref.u_conv4_weights.data()),
      engine_ref.get_buffer(app_data_ref.u_conv4_bias.data()),
      engine_ref.get_buffer(app_data_ref.u_conv4_out.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
      .input_height = model::kConv3OutHeight,
      .input_width = model::kConv3OutWidth,
      .weight_output_channels = model::kConv4OutChannels,
      .weight_input_channels = model::kConv3OutChannels,
      .weight_height = model::kKernelSize,
      .weight_width = model::kKernelSize,
      .bias_number_of_elements = model::kConv4BiasSize,
      .kernel_size = model::kKernelSize,
      .stride = model::kStride,
      .padding = model::kPadding,
      .output_height = model::kConv4OutHeight,
      .output_width = model::kConv4OutWidth,
      .relu = model::kRelu,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 7 (fifth conv2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage7(Sequence *seq) {
  const int total_iterations =
      model::kConv5OutChannels * model::kConv5OutHeight * model::kConv5OutWidth;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_conv4_out.data()),
      engine_ref.get_buffer(app_data_ref.u_conv5_weights.data()),
      engine_ref.get_buffer(app_data_ref.u_conv5_bias.data()),
      engine_ref.get_buffer(app_data_ref.u_conv5_out.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
      .input_height = model::kConv4OutHeight,
      .input_width = model::kConv4OutWidth,
      .weight_output_channels = model::kConv5OutChannels,
      .weight_input_channels = model::kConv4OutChannels,
      .weight_height = model::kKernelSize,
      .weight_width = model::kKernelSize,
      .bias_number_of_elements = model::kConv5BiasSize,
      .kernel_size = model::kKernelSize,
      .stride = model::kStride,
      .padding = model::kPadding,
      .output_height = model::kConv5OutHeight,
      .output_width = model::kConv5OutWidth,
      .relu = model::kRelu,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 8 (third maxpool2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage8(Sequence *seq) {
  const int total_iterations =
      model::kConv5OutChannels * model::kPool3OutHeight * model::kPool3OutWidth;

  auto algo = algorithms.at("maxpool2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_conv5_out.data()),
      engine_ref.get_buffer(app_data_ref.u_pool3_out.data()),
  });

  algo->update_push_constants(MaxpoolPushConstants{
      .input_channels = model::kConv5OutChannels,
      .input_height = model::kConv5OutHeight,
      .input_width = model::kConv5OutWidth,
      .pool_size = model::kPoolSize,
      .stride = model::kPoolStride,
      .output_height = model::kPool3OutHeight,
      .output_width = model::kPool3OutWidth,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 9 (linear)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage9(Sequence *seq) {
  const int total_iterations = model::kLinearOutFeatures;

  auto algo = algorithms.at("linear").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_pool3_out.data()),
      engine_ref.get_buffer(app_data_ref.u_linear_weights.data()),
      engine_ref.get_buffer(app_data_ref.u_linear_bias.data()),
      engine_ref.get_buffer(app_data_ref.u_linear_out.data()),
  });

  algo->update_push_constants(LinearPushConstants{
      .in_features = model::kLinearInFeatures,
      .out_features = model::kLinearOutFeatures,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();
  seq->sync();
}

}  // namespace vulkan
