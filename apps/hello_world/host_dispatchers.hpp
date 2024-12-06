#pragma once

#include "app_data.hpp"

#include <future>

#include "host_kernels.hpp"

namespace cpu {

[[nodiscard]] inline std::future<void>
run_stage1(const cuda::AppData &app_data) {
  return std::async(std::launch::async, [app_data]() {
    cpu::kernels::vector_add(app_data.input_a->data(), app_data.input_b->data(),
                             app_data.output->data(), 0, app_data.n);
  });
}

} // namespace cpu
