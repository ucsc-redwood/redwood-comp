#pragma once

#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1
#endif

#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif

#include <limits>
#include <memory>
#include <vulkan/vulkan.hpp>

#include "vk_mem_alloc.h"

extern VmaAllocator g_vma_allocator;

namespace vulkan {

/**
 * @brief Core Vulkan initialization and device management
 *
 * BaseEngine handles the fundamental Vulkan setup required for compute
 * operations:
 * - Instance creation and validation layer setup
 * - Physical device selection (preferring integrated GPUs)
 * - Logical device and compute queue creation
 * - VMA (Vulkan Memory Allocator) initialization
 *
 * This class serves as the foundation for the compute framework, providing
 * the basic Vulkan objects needed by other components.
 *
 * Example usage:
 * ```cpp
 * // Create engine with validation layers enabled
 * auto engine = std::make_unique<BaseEngine>(true);
 *
 * // Get device and queue for other operations
 * auto device = engine->get_device_ptr();
 * auto compute_queue = engine->get_compute_queue_ptr();
 * auto queue_family = engine->get_compute_queue_family_index();
 * ```
 */
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

  /**
   * @brief Get the logical device
   * @return Shared pointer to the Vulkan device
   */
  [[nodiscard]] std::shared_ptr<vk::Device> get_device_ptr() const {
    return std::make_shared<vk::Device>(device_);
  }

  /**
   * @brief Get the compute queue
   * @return Shared pointer to the compute queue
   */
  [[nodiscard]] std::shared_ptr<vk::Queue> get_compute_queue_ptr() const {
    return std::make_shared<vk::Queue>(compute_queue_);
  }

  /**
   * @brief Get the compute queue family index
   * @return Index of the compute queue family
   */
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
  // vk::detail::DynamicLoader dl_;

  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_;
  PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr_;

  std::vector<const char*> enabledLayers_;
};

}  // namespace vulkan
