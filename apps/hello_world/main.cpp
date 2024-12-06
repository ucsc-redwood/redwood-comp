#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <memory_resource>

#include "app_data.hpp"
#include "host/host_dispatchers.hpp"
#include "redwood/backends.hpp"

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

#include "cuda/device_dispatchers.cuh"
#include "redwood/cuda/cu_mem_resource.cuh"

void run_cuda_backend_demo(const size_t n) {
  cuda::CudaMemoryResource cuda_mr;
  AppData app_data(n, &cuda_mr);
  cuda::run_stage1(app_data);
  print_output(app_data);
}

#endif

#ifdef REDWOOD_VULKAN_BACKEND

#include "redwood/vulkan/vk_allocator.hpp"

void run_vulkan_backend_demo(const size_t n) {
  vulkan::VulkanMemoryResource vk_mr;
  AppData app_data(n, &vk_mr);
}

#endif

void run_cpu_backend_demo(const size_t n) {
  auto host_mr = std::pmr::new_delete_resource();
  AppData app_data(n, host_mr);

  cpu::run_stage1(app_data).wait();
  print_output(app_data);
}

int main(int argc, char** argv) {
  CLI::App app("Hello World");

  std::string device_id;
  app.add_option("-d,--device", device_id, "Device ID")->required();
  CLI11_PARSE(app, argc, argv);

  spdlog::set_level(spdlog::level::trace);

  constexpr auto n = 1024;

  if constexpr (is_backend_enabled(BackendType::kCPU)) {
    spdlog::info("CPU backend is enabled");
    run_cpu_backend_demo(n);
  }
  if constexpr (is_backend_enabled(BackendType::kCUDA)) {
    spdlog::info("CUDA backend is enabled");
    run_cuda_backend_demo(n);
  }
  if constexpr (is_backend_enabled(BackendType::kVulkan)) {
    spdlog::info("Vulkan backend is enabled");
    run_vulkan_backend_demo(n);
  }

  spdlog::info("Done");
  return EXIT_SUCCESS;
}
