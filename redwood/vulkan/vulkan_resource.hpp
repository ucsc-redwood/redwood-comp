#pragma once

#include <memory>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_NO_CONSTRUCTORS 1
#include <vulkan/vulkan.hpp>

namespace vulkan {

/**
 * @brief Base class for Vulkan resources providing RAII-style management
 *
 * VulkanResource is a template class that serves as a base for classes that
 * wrap Vulkan handles (like buffers, pipelines, shader modules, etc). It
 * provides:
 * - Shared device pointer management
 * - Common handle access methods
 * - Pure virtual destroy() method for cleanup
 *
 * @tparam HandleT The Vulkan handle type being wrapped (e.g., vk::Buffer,
 * vk::Pipeline)
 *
 * Example usage:
 * @code
 * class Buffer : public VulkanResource<vk::Buffer> {
 *     explicit Buffer(std::shared_ptr<vk::Device> device_ptr)
 *         : VulkanResource<vk::Buffer>(device_ptr) {
 *         // Create buffer...
 *     }
 *
 * protected:
 *     void destroy() override {
 *         if (handle_) {
 *             device_ptr_->destroyBuffer(handle_);
 *         }
 *     }
 * };
 * @endcode
 */
template <typename HandleT>
class VulkanResource {
 public:
  VulkanResource() = delete;

  /**
   * @brief Constructs a VulkanResource
   *
   * @param device_ptr Shared pointer to the Vulkan logical device
   */
  explicit VulkanResource(std::shared_ptr<vk::Device> device_ptr)
      : device_ptr_(std::move(device_ptr)) {}

  /**
   * @brief Get the underlying Vulkan handle
   * @return Reference to the Vulkan handle
   */
  [[nodiscard]] HandleT& get_handle() { return handle_; }

  /**
   * @brief Get the underlying Vulkan handle (const version)
   * @return Const reference to the Vulkan handle
   */
  [[nodiscard]] const HandleT& get_handle() const { return handle_; }

  virtual ~VulkanResource() = default;

 protected:
  /**
   * @brief Pure virtual function for resource cleanup
   *
   * Derived classes must implement this to properly destroy their Vulkan
   * resources. This should be called in the derived class destructor.
   */
  virtual void destroy() = 0;

  /** @brief Shared pointer to the Vulkan logical device */
  std::shared_ptr<vk::Device> device_ptr_;

  /** @brief The wrapped Vulkan handle */
  HandleT handle_;
};

}  // namespace vulkan
