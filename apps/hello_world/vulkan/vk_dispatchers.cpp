#include "vk_dispatchers.hpp"

#include <spdlog/spdlog.h>

namespace vulkan {

void run_stage1(Engine &engine, AppData &app_data) {
  spdlog::info("Running stage 1");

  //   auto seq = engine.sequence();
  struct PushConstants {
    uint32_t num_elements;
  };

  auto algo =
      engine
          .algorithm("hello_vector_add.comp",
                     {
                         engine.get_vk_buffer(app_data.u_input_a.data()),
                         engine.get_vk_buffer(app_data.u_input_b.data()),
                         engine.get_vk_buffer(app_data.u_output.data()),
                     })
          ->set_push_constants<PushConstants>(
              {static_cast<uint32_t>(app_data.n)})
          ->build();

  auto seq = engine.sequence();

  seq->record_commands(algo.get(), app_data.n);

  seq->launch_kernel_async();

  seq->sync();
}

}  // namespace vulkan