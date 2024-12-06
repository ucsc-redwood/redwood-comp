#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <memory_resource>

#include "app_data.hpp"

#ifdef REDWOOD_CUDA_BACKEND

#include "redwood/cuda/cu_mem_resource.cuh"

void run_cuda_backend_demo(const size_t n) {
  cuda::CudaMemoryResource cuda_mr;
  AppData app_data(n, &cuda_mr);
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
}

enum class BackendType { kCPU, kCUDA, kVulkan };

constexpr BackendType kBackendType = BackendType::kCUDA;

int main(int argc, char** argv) {
  CLI::App app("Hello World");

  std::string device_id;
  app.add_option("-d,--device", device_id, "Device ID")->required();
  CLI11_PARSE(app, argc, argv);

  spdlog::set_level(spdlog::level::trace);

  constexpr auto n = 1024;

  run_cpu_backend_demo(n);

#ifdef REDWOOD_CUDA_BACKEND
  run_cuda_backend_demo(n);
#endif

#ifdef REDWOOD_VULKAN_BACKEND
  run_vulkan_backend_demo(n);
#endif

  spdlog::info("Done");
  return EXIT_SUCCESS;
}
