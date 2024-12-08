#include <spdlog/spdlog.h>

#include <iostream>
#include <memory>

#include "algorithm.hpp"
#include "base_engine.hpp"
#include "sequence.hpp"
#include "vma_pmr.hpp"

class VulkanEngine final : public vulkan::BaseEngine {
 public:
  explicit VulkanEngine()
      : vulkan::BaseEngine(),
        mr_ptr(std::make_unique<VulkanMemoryResource>(get_device())) {}

  [[nodiscard]]
  std::shared_ptr<vulkan::Algorithm> algorithm(
      const std::string_view comp_name,
      const std::vector<vk::Buffer> &buffers) const {
    return std::make_shared<vulkan::Algorithm>(
        mr_ptr.get(), comp_name, buffers);
  }

  [[nodiscard]]
  std::shared_ptr<vulkan::Sequence> sequence() {
    return std::make_shared<vulkan::Sequence>(
        this->get_device(),
        this->get_compute_queue(),
        this->get_compute_queue_family_index());
  }

  [[nodiscard]]
  VulkanMemoryResource *get_mr() {
    return mr_ptr.get();
  }

  [[nodiscard]]
  vk::Buffer get_buffer(void *ptr) {
    return mr_ptr->get_buffer_from_pointer(ptr);
  }

 private:
  std::unique_ptr<VulkanMemoryResource> mr_ptr;
};

int main() {
  spdlog::set_level(spdlog::level::trace);

  VulkanEngine engine;

  const size_t n = 1024;
  std::pmr::vector<float> vec1(n, engine.get_mr());
  std::pmr::vector<float> vec2(n, engine.get_mr());
  std::pmr::vector<float> vec3(n, engine.get_mr());

  std::ranges::fill(vec1, 1.0f);
  std::ranges::fill(vec2, 2.0f);
  std::ranges::fill(vec3, 0.0f);

  vk::Buffer vec1_buffer = engine.get_buffer(vec1.data());
  vk::Buffer vec2_buffer = engine.get_buffer(vec2.data());
  vk::Buffer vec3_buffer = engine.get_buffer(vec3.data());

  struct PushConstants {
    int n;
  };

  auto algo = engine
                  .algorithm("hello_vector_add.comp",
                             {vec1_buffer, vec2_buffer, vec3_buffer})
                  ->set_push_constants<PushConstants>({n})
                  ->build();

  auto seq = engine.sequence();

  seq->record_commands(algo.get(), n);
  seq->launch_kernel_async();

  seq->sync();

  // print first 10 elements of vec3
  std::cout << "vec3: ";
  for (size_t i = 0; i < 10; ++i) {
    std::cout << vec3[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
