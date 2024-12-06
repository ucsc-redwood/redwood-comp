#include "redwood/cuda/cuda_engine.hpp"

#include <spdlog/spdlog.h>

int main() {
  spdlog::set_level(spdlog::level::trace);

  cuda::Engine engine;

  auto input_a = engine.typed_buffer<int>(1024);
  auto input_b = engine.typed_buffer<int>(1024);
  auto output = engine.typed_buffer<int>(1024);

  return EXIT_SUCCESS;
}
