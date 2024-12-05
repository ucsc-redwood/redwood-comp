
#include <memory>
#include <spdlog/spdlog.h>

#include "app_data.hpp"

// #include "cuda/cu_buffer_typed.cuh"

int main() {
  spdlog::set_level(spdlog::level::trace);

  auto buffer = std::make_unique<cuda::TypedBuffer<int>>(1024);

  std::ranges::fill(buffer->begin(), buffer->end(), 1);

  for (size_t i = 0; i < 10; ++i) {
    spdlog::info("buffer[{}] = {}", i, buffer->at(i));
  }

  app::demo::AppData app_data;

  return 0;
}
