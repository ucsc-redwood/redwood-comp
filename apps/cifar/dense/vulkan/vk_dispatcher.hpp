#pragma once

#include "../../app_data.hpp"
#include "redwood/vulkan/engine.hpp"

namespace vulkan {

struct Dispatcher {
  explicit Dispatcher(Engine &engine, AppData &app_data);

  void run_stage1(Sequence *seq);
  void run_stage2(Sequence *seq);
  void run_stage3(Sequence *seq);
  void run_stage4(Sequence *seq);
  void run_stage5(Sequence *seq);
  void run_stage6(Sequence *seq);
  void run_stage7(Sequence *seq);
  void run_stage8(Sequence *seq);
  void run_stage9(Sequence *seq);

  Engine &engine_ref;
  AppData &app_data_ref;
  std::unordered_map<std::string, std::shared_ptr<Algorithm>> algorithms;

  struct Conv2dPushConstants {
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

  struct MaxpoolPushConstants {
    uint32_t input_channels;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t pool_size;
    uint32_t stride;
    uint32_t output_height;
    uint32_t output_width;
  };

  struct LinearPushConstants {
    uint32_t in_features;
    uint32_t out_features;
  };
};

}  // namespace vulkan
