#include <spdlog/spdlog.h>

#include <iostream>
#include <memory>

#include "algorithm.hpp"
#include "base_engine.hpp"
#include "vma_pmr.hpp"

struct VulkanEngine {
  explicit VulkanEngine()
      : base_engine(),
        mr_ptr(
            std::make_unique<VulkanMemoryResource>(base_engine.get_device())) {}

  [[nodiscard]]
  std::shared_ptr<vulkan::Algorithm> algorithm(
      const std::string_view comp_name,
      const std::vector<vk::Buffer> &buffers) const {
    return std::make_shared<vulkan::Algorithm>(
        mr_ptr.get(), comp_name, buffers);
  }

  vulkan::BaseEngine base_engine;
  std::unique_ptr<VulkanMemoryResource> mr_ptr;
};

int main() {
  spdlog::set_level(spdlog::level::trace);

  VulkanEngine engine;

  const size_t n = 1024;
  std::pmr::vector<float> vec1(n, engine.mr_ptr.get());
  std::pmr::vector<float> vec2(n, engine.mr_ptr.get());
  std::pmr::vector<float> vec3(n, engine.mr_ptr.get());

  std::cout << "vec1 size address: " << (void *)vec1.data() << std::endl;
  std::cout << "vec2 size address: " << (void *)vec2.data() << std::endl;

  std::ranges::fill(vec1, 1.0f);
  std::ranges::fill(vec2, 2.0f);
  std::ranges::fill(vec3, 0.0f);

  vk::Buffer vec1_buffer = engine.mr_ptr->get_buffer_from_pointer(vec1.data());
  vk::Buffer vec2_buffer = engine.mr_ptr->get_buffer_from_pointer(vec2.data());
  vk::Buffer vec3_buffer = engine.mr_ptr->get_buffer_from_pointer(vec3.data());

  struct PushConstants {
    int n;
  };

  auto algo = engine
                  .algorithm("hello_vector_add.comp",
                             {vec1_buffer, vec2_buffer, vec3_buffer})
                  ->set_push_constants<PushConstants>({n})
                  ->build();

  // auto algo = std::make_shared<vulkan::Algorithm>(
  //                 engine.mr,
  //                 "hello_vector_add.comp",
  //                 std::vector{vec1_buffer, vec2_buffer, vec3_buffer})
  //                 ->set_push_constants<PushConstants>({n})
  //                 ->build();

  // auto algo_ptr = std::shared_ptr<vulkan::Algorithm>(&algo);

  // algo_ptr->set_push_constants<PushConstants>({n})->build();

  // auto algo = engine
  // auto algo = engine
  //                 .algorithm("hello_vector_add.comp",
  //                            std::vector{vec1_buffer, vec2_buffer,
  //                            vec3_buffer})
  //                 ->set_push_constants<PushConstants>({n})
  //                 ->build();

  return 0;
}
