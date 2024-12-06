#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include "app_data.hpp"
#include "device_dispatchers.cuh"
#include "host_dispatchers.hpp"

#include <algorithm>

int main(int argc, char **argv) {
  CLI::App app("Hello World");

  bool use_cuda = false;
  app.add_flag("--cuda", use_cuda, "Use CUDA");

  CLI11_PARSE(app, argc, argv);

  spdlog::set_level(spdlog::level::trace);

  cuda::Engine engine;

  cuda::AppData app_data(engine, 1024);

  std::ranges::fill(*app_data.input_a, 1);
  std::ranges::fill(*app_data.input_b, 2);

  if (use_cuda) {
    cuda::run_stage1(app_data);
  } else {
    cpu::run_stage1(app_data).wait();
  }

  // print the first 10 elements of output
  for (size_t i = 0; i < 10; ++i) {
    spdlog::info("output[{}] = {}", i, app_data.output->at(i));
  }

  cuda::run_stage1(app_data);

  return EXIT_SUCCESS;
}
