#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <algorithm>

#include "app_data.hpp"
#include "device_dispatchers.cuh"
#include "host_dispatchers.hpp"

int main(int argc, char **argv) {
  CLI::App app("Hello World");

  bool use_cuda = false;
  app.add_flag("--cuda", use_cuda, "Use CUDA");
  CLI11_PARSE(app, argc, argv);

  spdlog::set_level(spdlog::level::trace);

  // Initialize compute engine and application data
  cuda::Engine engine;

  cuda::AppData app_data(engine, 1024);

  std::ranges::fill(*app_data.input_a, 1);
  std::ranges::fill(*app_data.input_b, 2);

  // Run the stage 1 of the application
  if (use_cuda) {
    cuda::run_stage1(engine, app_data);
    engine.sync();
  } else {
    cpu::run_stage1(app_data).wait();
  }

  // print the first 10 elements of output
  for (size_t i = 0; i < 10; ++i) {
    spdlog::info("output[{}] = {}", i, app_data.output->at(i));
  }

  return EXIT_SUCCESS;
}
