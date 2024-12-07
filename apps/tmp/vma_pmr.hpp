#pragma once

#include <memory_resource>
#include <mutex>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unordered_map>

#include <vk_mem_alloc.h>

#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1
#endif


#include <vulkan/vulkan.hpp>

// Externally defined VMA allocator (you must have this created and initialized
// somewhere)
extern VmaAllocator g_vma_allocator;

// A small helper macro to check Vulkan results
inline void CHECK_VK_RESULT(VkResult result, const char *msg) {
  if (result != VK_SUCCESS) {
    spdlog::error("Vulkan error: {} - {}",
                  vk::to_string(static_cast<vk::Result>(result)), msg);
    throw std::runtime_error("Vulkan Error");
  }
}

// Structure to keep track of the buffer allocation details
struct VulkanAllocationRecord {
  VkBuffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo allocInfo;
};

// ----------------------------------------------------------------------------
// VulkanMemoryResource
// ----------------------------------------------------------------------------

class VulkanMemoryResource : public std::pmr::memory_resource {
public:
  // We use the requested defaults for usage flags and allocation flags.
  VulkanMemoryResource(vk::Device device,
                       vk::BufferUsageFlags buffer_usage =
                           vk::BufferUsageFlagBits::eStorageBuffer,
                       VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO,
                       VmaAllocationCreateFlags flags =
                           VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT);

  ~VulkanMemoryResource();

  [[nodiscard]]
  vk::Buffer get_buffer_from_pointer(void *p);

  [[nodiscard]]
  vk::DescriptorBufferInfo get_descriptor_buffer_info(void *p);

protected:
  void *do_allocate(std::size_t bytes, std::size_t alignment) override;

  void do_deallocate(void *p, std::size_t bytes,
                     std::size_t alignment) override;

  bool
  do_is_equal(const std::pmr::memory_resource &other) const noexcept override;

private:
  vk::Device device_;
  vk::BufferUsageFlags bufferUsage_;
  VmaMemoryUsage memoryUsage_;
  VmaAllocationCreateFlags allocationFlags_;

  mutable std::mutex mutex_;
  std::unordered_map<void *, VulkanAllocationRecord> allocations_;
};
