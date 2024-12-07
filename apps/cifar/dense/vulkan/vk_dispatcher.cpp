#include "vk_dispatcher.hpp"

#include <cstdint>

#include "apps/cifar/dense/app_data.hpp"

namespace vulkan {

void run_stage1(Engine &engine, AppData &app_data) {
  spdlog::info("Running stage 1");

  // layout(push_constant) uniform Params {
  //   uint input_height;
  //   uint input_width;
  //   uint weight_output_channels;
  //   uint weight_input_channels;
  //   uint weight_height;
  //   uint weight_width;
  //   uint bias_number_of_elements;
  //   uint kernel_size;
  //   uint stride;
  //   uint padding;
  //   uint output_height;
  //   uint output_width;
  //   bool relu;
  // }
  // params;

  struct PushConstants {
    uint32_t input_height;
    uint32_t input_width;
    uint32_t weight_output_channels;
    uint32_t weight_input_channels;
    uint32_t weight_height;
    uint32_t weight_width;
    uint32_t bias_number_of_elements;
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;
    uint32_t output_height;
    uint32_t output_width;
    bool relu;
  };

  // layout(local_size_x = 256) in;

  // layout(std430, set = 0, binding = 0) readonly buffer InputBuffer {
  //   float input_data[];
  // };

  // layout(std430, set = 0, binding = 1) readonly buffer WeightBuffer {
  //   float weight_data[];
  // };

  // layout(std430, set = 0, binding = 2) readonly buffer BiasBuffer {
  //   float bias_data[];
  // };

  // layout(std430, set = 0, binding = 3) writeonly buffer OutputBuffer {
  //   float output_data[];
  // };

  static auto algo =
      engine
          .algorithm("cifar_conv2d.comp",
                     {
                         engine.get_vk_buffer(app_data.u_conv1_out.data()),
                         engine.get_vk_buffer(app_data.u_conv2_weights.data()),
                         engine.get_vk_buffer(app_data.u_conv2_bias.data()),
                         engine.get_vk_buffer(app_data.u_conv2_out.data()),
                     })
          ->set_push_constants<PushConstants>({
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

  auto seq = engine.sequence();

  const int total_iterations =
      model::kConv1OutChannels * model::kConv1OutHeight * model::kConv1OutWidth;

  seq->record_commands(algo.get(), total_iterations);

  seq->launch_kernel_async();

  seq->sync();
}

}  // namespace vulkan
