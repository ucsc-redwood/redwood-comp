#pragma once

#include "algorithm.hpp"
#include "base_engine.hpp"
#include "sequence.hpp"
#include "vma_pmr.hpp"

namespace vulkan {

class Engine final : public vulkan::BaseEngine {
 public:
  explicit Engine()
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

}  // namespace vulkan
