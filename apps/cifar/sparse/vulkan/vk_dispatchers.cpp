#include "vk_dispatchers.hpp"

namespace vulkan {

// Input Image dimensions
constexpr int kInputChannels = 3;
constexpr int kInputHeight = 32;
constexpr int kInputWidth = 32;

// Convolution parameters
constexpr int kKernelSize = 3;
constexpr int kStride = 1;
constexpr int kPadding = 1;

// Pooling parameters
constexpr int kPoolSize = 2;
constexpr int kPoolStride = 2;

constexpr bool kRelu = true;

Dispatcher::Dispatcher(Engine &engine, AppData &app_data)
    : engine_ref(engine), app_data_ref(app_data) {
  auto conv2d_algo =
      engine
          .algorithm("cifar_sparse_conv2d.comp",
                     {
                         // We still need the buffer here, because we need to
                         // know the size to setup the vk::Pipeline. While the
                         // values here does not matter yet.
                         engine.get_buffer(app_data.u_image_data.data()),
                         engine.get_buffer(app_data.u_conv1_values.data()),
                         engine.get_buffer(app_data.u_conv1_row_ptr.data()),
                         engine.get_buffer(app_data.u_conv1_col_idx.data()),
                         engine.get_buffer(app_data.u_conv1_bias.data()),
                         engine.get_buffer(app_data.u_conv1_output.data()),
                     })
          ->set_push_constants<Conv2dPushConstants>({
              // Similarly here, we need to know how many elements we have in
              .input_height = 0,
              .input_width = 0,
              .weight_output_channels = 0,
              .weight_input_channels = 0,
              .weight_height = 0,
              .weight_width = 0,
              .kernel_size = 0,
              .stride = 0,
              .padding = 0,
              .relu = false,
          })
          ->build();

  algorithms.try_emplace("conv2d", std::move(conv2d_algo));

  auto maxpool2d_algo =
      engine
          .algorithm(
              "cifar_sparse_maxpool.comp",
              {
                  engine.get_buffer(app_data.u_conv1_output.data()),  // input
                  engine.get_buffer(app_data.u_pool1_output.data()),  // output
              })
          ->set_push_constants<MaxpoolPushConstants>({
              .input_channels = 0,
              .input_height = 0,
              .input_width = 0,
              .pool_size = 0,
              .stride = 0,
          })
          ->build();

  algorithms.try_emplace("maxpool2d", std::move(maxpool2d_algo));

  auto linear_algo =
      engine
          .algorithm(
              "cifar_sparse_linear.comp",
              {
                  engine.get_buffer(app_data.u_pool3_output.data()),  // input
                  engine.get_buffer(
                      app_data.u_linear_values.data()),  // weights
                  engine.get_buffer(
                      app_data.u_linear_row_ptr.data()),  // row ptr
                  engine.get_buffer(
                      app_data.u_linear_col_idx.data()),             // col idx
                  engine.get_buffer(app_data.u_linear_bias.data()),  // bias
                  engine.get_buffer(app_data.u_linear_output.data()),  // output
              })
          ->set_push_constants<LinearPushConstants>({
              .weight_matrix_rows = 0,
              .weight_matrix_cols = 0,
          })
          ->build();

  algorithms.try_emplace("linear", std::move(linear_algo));
}

// ----------------------------------------------------------------------------
// Stage 1 (first conv2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage1(Sequence *seq, const bool sync) {
  const auto total_iterations = app_data_ref.conv1_weights.rows;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_image_data.data()),
      engine_ref.get_buffer(app_data_ref.u_conv1_values.data()),
      engine_ref.get_buffer(app_data_ref.u_conv1_row_ptr.data()),
      engine_ref.get_buffer(app_data_ref.u_conv1_col_idx.data()),
      engine_ref.get_buffer(app_data_ref.u_conv1_bias.data()),
      engine_ref.get_buffer(app_data_ref.u_conv1_output.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
      .input_height = kInputHeight,
      .input_width = kInputWidth,
      .weight_output_channels = 64,
      .weight_input_channels = kInputChannels,
      .weight_height = static_cast<uint32_t>(app_data_ref.conv1_weights.rows),
      .weight_width = static_cast<uint32_t>(app_data_ref.conv1_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  if (sync) {
    seq->sync();
  }
}

// ----------------------------------------------------------------------------
// Stage 2 (first maxpool2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage2(Sequence *seq, const bool sync) {
  constexpr auto output_height = (kInputHeight - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (kInputWidth - kPoolSize) / kPoolStride + 1;
  auto total_iterations = kInputChannels * output_height * output_width;

  auto algo = algorithms.at("maxpool2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_conv1_output.data()),
      engine_ref.get_buffer(app_data_ref.u_pool1_output.data()),
  });

  algo->update_push_constants(MaxpoolPushConstants{
      .input_channels = 64,
      .input_height = 32,
      .input_width = 32,
      .pool_size = kPoolSize,
      .stride = kPoolStride,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  if (sync) {
    seq->sync();
  }
}

// ----------------------------------------------------------------------------
// Stage 3 (second conv2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage3(Sequence *seq, const bool sync) {
  const auto total_iterations = app_data_ref.conv2_weights.rows;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_pool1_output.data()),
      engine_ref.get_buffer(app_data_ref.u_conv2_values.data()),
      engine_ref.get_buffer(app_data_ref.u_conv2_row_ptr.data()),
      engine_ref.get_buffer(app_data_ref.u_conv2_col_idx.data()),
      engine_ref.get_buffer(app_data_ref.u_conv2_bias.data()),
      engine_ref.get_buffer(app_data_ref.u_conv2_output.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
      .input_height = 16,
      .input_width = 16,
      .weight_output_channels = 192,
      .weight_input_channels = 64,
      .weight_height = static_cast<uint32_t>(app_data_ref.conv2_weights.rows),
      .weight_width = static_cast<uint32_t>(app_data_ref.conv2_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  if (sync) {
    seq->sync();
  }
}

//----------------------------------------------------------------------------
// Stage 4 (second maxpool2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage4(Sequence *seq, const bool sync) {
  constexpr auto input_channels = 192;
  constexpr auto input_height = 16;
  constexpr auto input_width = 16;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations =
      input_channels * output_height * output_width;

  auto algo = algorithms.at("maxpool2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_conv2_output.data()),
      engine_ref.get_buffer(app_data_ref.u_pool2_output.data()),
  });

  algo->update_push_constants(MaxpoolPushConstants{
      .input_channels = input_channels,
      .input_height = input_height,
      .input_width = input_width,
      .pool_size = kPoolSize,
      .stride = kPoolStride,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  if (sync) {
    seq->sync();
  }
}

// ----------------------------------------------------------------------------
// Stage 5 (third conv2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage5(Sequence *seq, const bool sync) {
  const auto total_iterations = app_data_ref.conv3_weights.rows;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_pool2_output.data()),
      engine_ref.get_buffer(app_data_ref.u_conv3_values.data()),
      engine_ref.get_buffer(app_data_ref.u_conv3_row_ptr.data()),
      engine_ref.get_buffer(app_data_ref.u_conv3_col_idx.data()),
      engine_ref.get_buffer(app_data_ref.u_conv3_bias.data()),
      engine_ref.get_buffer(app_data_ref.u_conv3_output.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
      .input_height = 8,
      .input_width = 8,
      .weight_output_channels = 384,
      .weight_input_channels = 192,
      .weight_height = static_cast<uint32_t>(app_data_ref.conv3_weights.rows),
      .weight_width = static_cast<uint32_t>(app_data_ref.conv3_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  if (sync) {
    seq->sync();
  }
}

// ----------------------------------------------------------------------------
// Stage 6 (fourth conv2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage6(Sequence *seq, const bool sync) {
  const auto total_iterations = app_data_ref.conv4_weights.rows;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_conv3_output.data()),
      engine_ref.get_buffer(app_data_ref.u_conv4_values.data()),
      engine_ref.get_buffer(app_data_ref.u_conv4_row_ptr.data()),
      engine_ref.get_buffer(app_data_ref.u_conv4_col_idx.data()),
      engine_ref.get_buffer(app_data_ref.u_conv4_bias.data()),
      engine_ref.get_buffer(app_data_ref.u_conv4_output.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
      .input_height = 8,
      .input_width = 8,
      .weight_output_channels = 256,
      .weight_input_channels = 384,
      .weight_height = static_cast<uint32_t>(app_data_ref.conv4_weights.rows),
      .weight_width = static_cast<uint32_t>(app_data_ref.conv4_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  if (sync) {
    seq->sync();
  }
}

// ----------------------------------------------------------------------------
// Stage 7 (fifth conv2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage7(Sequence *seq, const bool sync) {
  const auto total_iterations = app_data_ref.conv5_weights.rows;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_conv4_output.data()),
      engine_ref.get_buffer(app_data_ref.u_conv5_values.data()),
      engine_ref.get_buffer(app_data_ref.u_conv5_row_ptr.data()),
      engine_ref.get_buffer(app_data_ref.u_conv5_col_idx.data()),
      engine_ref.get_buffer(app_data_ref.u_conv5_bias.data()),
      engine_ref.get_buffer(app_data_ref.u_conv5_output.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
      .input_height = 8,
      .input_width = 8,
      .weight_output_channels = 256,
      .weight_input_channels = 256,
      .weight_height = static_cast<uint32_t>(app_data_ref.conv5_weights.rows),
      .weight_width = static_cast<uint32_t>(app_data_ref.conv5_weights.cols),
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  if (sync) {
    seq->sync();
  }
}

// ----------------------------------------------------------------------------
// Stage 8 (third maxpool2d)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage8(Sequence *seq, const bool sync) {
  constexpr auto input_channels = 256;
  constexpr auto input_height = 8;
  constexpr auto input_width = 8;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations =
      input_channels * output_height * output_width;

  auto algo = algorithms.at("maxpool2d").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_conv5_output.data()),
      engine_ref.get_buffer(app_data_ref.u_pool3_output.data()),
  });

  algo->update_push_constants(MaxpoolPushConstants{
      .input_channels = input_channels,
      .input_height = input_height,
      .input_width = input_width,
      .pool_size = kPoolSize,
      .stride = kPoolStride,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  if (sync) {
    seq->sync();
  }
}

// ----------------------------------------------------------------------------
// Stage 9 (linear)
// ----------------------------------------------------------------------------

void Dispatcher::run_stage9(Sequence *seq, const bool sync) {
  const auto total_iterations = app_data_ref.linear_weights.rows;

  auto algo = algorithms.at("linear").get();

  algo->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_pool3_output.data()),
      engine_ref.get_buffer(app_data_ref.u_linear_values.data()),
      engine_ref.get_buffer(app_data_ref.u_linear_row_ptr.data()),
      engine_ref.get_buffer(app_data_ref.u_linear_col_idx.data()),
      engine_ref.get_buffer(app_data_ref.u_linear_bias.data()),
      engine_ref.get_buffer(app_data_ref.u_linear_output.data()),
  });

  algo->update_push_constants(LinearPushConstants{
      .weight_matrix_rows =
          static_cast<uint32_t>(app_data_ref.linear_weights.rows),
      .weight_matrix_cols =
          static_cast<uint32_t>(app_data_ref.linear_weights.cols),
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  if (sync) {
    seq->sync();
  }
}

}  // namespace vulkan
