#include "vulkan/vk_engine.hpp"

#include "cuda/cu_engine.cuh"

#include <spdlog/spdlog.h>

int main() {
  spdlog::set_level(spdlog::level::trace);

  VulkanEngine engine;
  CudaEngine cu_engine;
  
  return 0;
}
