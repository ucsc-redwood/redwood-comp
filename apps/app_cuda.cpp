
#include <numeric>
#include <spdlog/spdlog.h>

#include "../redwood/cuda/cuda_engine.hpp"

int main() {
  cuda::Engine engine;

  spdlog::set_level(spdlog::level::trace);

  auto buffer_a = engine.typed_buffer<int>(1024)->random();
  auto buffer_b = engine.typed_buffer<float>(1024)->zeros();

  for (size_t i = 0; i < 10; ++i) {
    spdlog::info("buffer_a[{}] = {}", i, buffer_a->at(i));
    spdlog::info("buffer_b[{}] = {}", i, buffer_b->at(i));
  }

  return 0;
}
