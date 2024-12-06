#pragma once

#include <vk_mem_alloc.h>

#include "vulkan_resource.hpp"

namespace vulkan {

class Buffer;

using BufferVec = std::vector<std::shared_ptr<Buffer>>;

/**
 * @brief Buffer represents a Vulkan buffer with automatic memory management
 *
 * The Buffer class provides a high-level wrapper around Vulkan buffers with:
 * - Automatic memory allocation using VMA (Vulkan Memory Allocator)
 * - Automatic mapping for host-visible memory
 * - RAII-style resource management
 *
 * This class serves as the base for TypedBuffer<T> which provides type-safe
 * access. Typically used for storage buffers in compute shaders.
 *
 * Example usage:
 * ```cpp
 * // Create a 1MB storage buffer
 * auto buffer = std::make_shared<Buffer>(
 *     device,
 *     1024 * 1024,                          // size in bytes
 *     vk::BufferUsageFlagBits::eStorageBuffer,  // usage
 *     VMA_MEMORY_USAGE_AUTO,                // let VMA decide memory type
 *     VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
 *     VMA_ALLOCATION_CREATE_MAPPED_BIT       // CPU accessible and mapped
 * );
 *
 * // Zero initialize
 * buffer->zeros();
 *
 * // Access the mapped memory
 * float* data = buffer->as<float>();
 * data[0] = 1.0f;
 * ```
 */
class Buffer : public VulkanResource<vk::Buffer> {
 public:
  /**
   * @brief Constructs a Buffer instance
   *
   * @param device_ptr Vulkan logical device
   * @param size Size of the buffer in bytes
   * @param buffer_usage Vulkan buffer usage flags (e.g. storage, uniform)
   * @param memory_usage VMA memory usage hint
   * @param flags VMA allocation flags for controlling memory properties
   */
  explicit Buffer(std::shared_ptr<vk::Device> device_ptr,
                  vk::DeviceSize size,
                  vk::BufferUsageFlags buffer_usage =
                      vk::BufferUsageFlagBits::eStorageBuffer,
                  VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO,
                  VmaAllocationCreateFlags flags =
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                      VMA_ALLOCATION_CREATE_MAPPED_BIT);

  ~Buffer() override { destroy(); }

  /**
   * @brief Get typed pointer to the mapped memory
   *
   * @tparam T Type to interpret the memory as
   * @return Pointer to the mapped memory as type T
   */
  template <typename T>
  [[nodiscard]] T* as() {
    return reinterpret_cast<T*>(mapped_data_);
  }

  /**
   * @brief Zero initialize the entire buffer
   */
  void zeros() { std::memset(mapped_data_, 0, size_); }

  // --------------------------------------------------------------------------
  // Getters
  // --------------------------------------------------------------------------

  [[nodiscard]] vk::DeviceSize get_size_in_bytes() const { return size_; }

 protected:
  void destroy() override;

  /**
   * @brief Get descriptor info for the buffer
   * Used internally by Algorithm for descriptor set updates
   */
  [[nodiscard]] vk::DescriptorBufferInfo get_descriptor_buffer_info() const {
    return vk::DescriptorBufferInfo{
        .buffer = this->get_handle(),
        .offset = 0,
        .range = this->size_,
    };
  }

 private:
  // Vulkan Memory Allocator components
  VmaAllocation allocation_;
  vk::DeviceMemory memory_;
  vk::DeviceSize size_;

  // Raw pointer to the mapped data, CPU/GPU shared memory.
  std::byte* mapped_data_;

  friend class Engine;
  friend class Algorithm;
};

}  // namespace vulkan