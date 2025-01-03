#pragma once

#include <limits>

#include "vk.hpp"

namespace vulkan {

class BaseEngine {
 public:
  /**
   * @brief Constructs and initializes the Vulkan environment
   *
   * @param enable_validation_layer Enable Vulkan validation layers for
   * debugging
   */
  explicit BaseEngine(bool enable_validation_layer = true);

  ~BaseEngine() { destroy(); }

  [[nodiscard]] vk::Device &get_device() { return device_; }
  [[nodiscard]] vk::Queue &get_compute_queue() { return compute_queue_; }
  [[nodiscard]] uint32_t get_compute_queue_family_index() const {
    return compute_queue_family_index_;
  }

  /**
   * @brief Get the subgroup size
   * @return Subgroup size
   */
  [[nodiscard]] uint32_t get_subgroup_size() const;

 protected:
  void destroy() const;

  void initialize_dynamic_loader();
  void request_validation_layer();

  void create_instance();
  void create_physical_device(
      vk::PhysicalDeviceType type = vk::PhysicalDeviceType::eIntegratedGpu);
  void create_device(vk::QueueFlags queue_flags = vk::QueueFlagBits::eCompute);

  void initialize_vma_allocator();

  // Core vulkan handles
  vk::Instance instance_;
  vk::PhysicalDevice physical_device_;
  vk::Device device_;
  vk::Queue compute_queue_;

  uint32_t compute_queue_family_index_ = std::numeric_limits<uint32_t>::max();

 private:
  vk::DynamicLoader dl_;
  vk::DispatchLoaderDynamic dldi_;

  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_;
  //   PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr_;

  std::vector<const char *> enabledLayers_;
};

}  // namespace vulkan
