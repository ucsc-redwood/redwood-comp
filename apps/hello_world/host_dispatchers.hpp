#pragma once

#include <future>

#include "app_data.hpp"
#include "host_kernels.hpp"

namespace cpu {

[[nodiscard]] inline std::future<void> run_stage1(const AppData &app_data) {
  spdlog::debug("CPU kernel 'vector_add', n = {}", app_data.n);

  return std::async(std::launch::async, [&]() {
    cpu::kernels::vector_add(app_data.input_a->data(),
                             app_data.input_b->data(),
                             app_data.output->data(),
                             0,
                             app_data.n);
  });
}

}  // namespace cpu
