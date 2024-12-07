#include <spdlog/spdlog.h>

#include <iostream>
#include <memory>

#include "algorithm.hpp"
#include "base_engine.hpp"
#include "vma_pmr.hpp"

int main() {
  spdlog::set_level(spdlog::level::trace);

  vulkan::BaseEngine engine;
  VulkanMemoryResource mr(engine.get_device());

  const size_t n = 1024;
  std::pmr::vector<float> vec1(n, &mr);
  std::pmr::vector<float> vec2(n, &mr);
  std::pmr::vector<float> vec3(n, &mr);

  std::cout << "vec1 size address: " << (void *)vec1.data() << std::endl;
  std::cout << "vec2 size address: " << (void *)vec2.data() << std::endl;

  std::ranges::fill(vec1, 1.0f);
  std::ranges::fill(vec2, 2.0f);
  std::ranges::fill(vec3, 0.0f);

  vk::Buffer vec1_buffer = mr.get_buffer_from_pointer(vec1.data());
  vk::Buffer vec2_buffer = mr.get_buffer_from_pointer(vec2.data());
  vk::Buffer vec3_buffer = mr.get_buffer_from_pointer(vec3.data());

  struct PushConstants {
    int n;
  };

  auto algo = std::make_shared<vulkan::Algorithm>(
                  engine.get_device(),
                  mr,
                  "hello_vector_add.comp",
                  std::vector{vec1_buffer, vec2_buffer, vec3_buffer})
                  ->set_push_constants<PushConstants>({n})
                  ->build();

  return 0;
}
