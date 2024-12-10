#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <memory_resource>

#include "../app.hpp"
#include "app_data.hpp"
#include "host/host_dispatchers.hpp"
#include "redwood/backends.hpp"
#include "redwood/host/thread_pool.hpp"

// forward declare
void run_cpu_backend_demo(size_t n);
void run_cuda_backend_demo(size_t n);
void run_vulkan_backend_demo(size_t n);

// print the first 10 elements of the output
void print_output(const AppData& app_data) {
  for (size_t i = 0; i < 10; ++i) {
    spdlog::info("output[{}] = {}", i, app_data.u_output[i]);
  }
}

#ifdef REDWOOD_CUDA_BACKEND

#include <cuda_runtime_api.h>

#include "cuda/device_dispatchers.cuh"
#include "redwood/cuda/cu_dispatcher.cuh"
#include "redwood/cuda/cu_mem_resource.cuh"

void run_cuda_backend_demo(size_t n) {
  auto mr = std::make_shared<cuda::CudaMemoryResource>();

  AppData app_data(mr.get(), n);
  cuda::CuDispatcher disp(mr, 1);

  cuda::run_stage1(app_data, disp.stream(0));
  disp.sync(0);

  print_output(app_data);
}

#endif

#ifdef REDWOOD_VULKAN_BACKEND

#include "redwood/vulkan/engine.hpp"
#include "vulkan/vk_dispatchers.hpp"

void run_vulkan_backend_demo(const size_t n) {
  vulkan::Engine engine;
  AppData app_data(engine.get_mr(), n);

  vulkan::run_stage1(engine, app_data);

  print_output(app_data);
}

#endif

void run_cpu_backend_demo(const size_t n) {
  auto host_mr = std::pmr::new_delete_resource();
  AppData app_data(host_mr, n);

  core::thread_pool pool(g_small_cores);
  cpu::v2::run_stage1(app_data, pool, g_small_cores.size()).wait();

  print_output(app_data);
}

int main(int argc, char** argv) {
  INIT_APP("hello_world")

  constexpr auto n = 1024;

  run_cpu_backend_demo(n);

  if constexpr (is_backend_enabled(BackendType::kCUDA)) {
    run_cuda_backend_demo(n);
  }
  if constexpr (is_backend_enabled(BackendType::kVulkan)) {
    run_vulkan_backend_demo(n);
  }

  spdlog::info("Done");
  return EXIT_SUCCESS;
}
