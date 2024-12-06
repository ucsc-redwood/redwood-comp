#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <memory_resource>

#include "redwood/cuda/cu_mem_resource.cuh"

#include "redwood/vulkan/engine.hpp"
#include "redwood/vulkan/vk_allocator.hpp"

int main(int argc, char** argv) {
  CLI::App app("Hello World");

  std::string device_id;
  app.add_option("-d,--device", device_id, "Device ID")->required();
  CLI11_PARSE(app, argc, argv);

  spdlog::set_level(spdlog::level::trace);

  vulkan::Engine engine;

  constexpr auto n = 1024;

  auto host_mr = std::pmr::new_delete_resource();
  std::pmr::vector<int> v_cpu1(n, host_mr);
  std::pmr::vector<int> v_cpu2(n, host_mr);

  CudaMemoryResource cuda_mr;
  std::pmr::vector<int> v_cuda1(n, &cuda_mr);
  std::pmr::vector<int> v_cuda2(n, &cuda_mr);

  vulkan::VulkanMemoryResource vk_mr(engine);
  std::pmr::vector<int> v_vk1(n, &vk_mr);

  vulkan::VulkanMemoryResource vk_mr2(engine);
  std::pmr::vector<int> v_vk2(n, &vk_mr2);

  std::ranges::fill(v_cpu1, 1);
  std::ranges::fill(v_cpu2, 2);
  std::ranges::fill(v_cuda1, 3);
  std::ranges::fill(v_cuda2, 4);
  std::ranges::fill(v_vk1, 5);
  std::ranges::fill(v_vk2, 6);

  // print cpu vectors
  for (size_t i = 0; i < 10; ++i) {
    std::cout << v_cpu1[i] << " " << v_cpu2[i] << std::endl;
  }

  // print cuda vectors
  for (size_t i = 0; i < n; ++i) {
    std::cout << v_cuda1[i] << " " << v_cuda2[i] << std::endl;
  }

  // print vulkan vectors
  for (size_t i = 0; i < 10; ++i) {
    std::cout << v_vk1[i] << " " << v_vk2[i] << std::endl;
  }

  spdlog::info("Done");
  return EXIT_SUCCESS;
}
