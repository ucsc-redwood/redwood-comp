#include "app_data.hpp"
#include <spdlog/spdlog.h>

#include <algorithm>

int main() {
  spdlog::set_level(spdlog::level::trace);

  cuda::Engine engine;

  cuda::AppData app_data(engine, 1024);

  std::ranges::fill(*app_data.input_a, 1);
  std::ranges::fill(*app_data.input_b, 2);

  // print the first 10 elements of output
  for (size_t i = 0; i < 10; ++i) {
    spdlog::info("output[{}] = {}", i, app_data.output->at(i));
  }

  return EXIT_SUCCESS;
}
