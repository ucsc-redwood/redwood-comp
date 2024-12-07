#pragma once

#include <memory_resource>
#include <mutex>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unordered_map>

// #define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

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

// This is your custom Vulkan memory resource that uses VMA under the hood.
class VulkanMemoryResource : public std::pmr::memory_resource {
public:
  // We use the requested defaults for usage flags and allocation flags.
  VulkanMemoryResource(vk::Device device,
                       vk::BufferUsageFlags buffer_usage =
                           vk::BufferUsageFlagBits::eStorageBuffer,
                       VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO,
                       VmaAllocationCreateFlags flags =
                           VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT)
      : device_(device), bufferUsage_(buffer_usage), memoryUsage_(memory_usage),
        allocationFlags_(flags) {
    SPDLOG_DEBUG("VulkanMemoryResource created with usageFlags = {}, "
                 "memoryUsage = {}, allocFlags = {}",
                 (VkBufferUsageFlags)bufferUsage_, (int)memoryUsage_,
                 (int)allocationFlags_);
  }

  ~VulkanMemoryResource() {
    SPDLOG_DEBUG("VulkanMemoryResource destroyed");
    // Ideally, you should not have any outstanding allocations at this point.
    // If you do, that would be a logical error in your code.
  }

protected:
  // Allocate a buffer and return a pointer to mapped host memory.
  void *do_allocate(std::size_t bytes,
                    std::size_t alignment) override;

  // Deallocate the buffer by looking up our record.
  void do_deallocate(void *p, std::size_t bytes, std::size_t alignment) override;

  bool
  do_is_equal(const std::pmr::memory_resource &other) const noexcept override {
    return dynamic_cast<const VulkanMemoryResource *>(&other) != nullptr;
  }

private:
  vk::Device device_;
  vk::BufferUsageFlags bufferUsage_;
  VmaMemoryUsage memoryUsage_;
  VmaAllocationCreateFlags allocationFlags_;

  mutable std::mutex mutex_;
  std::unordered_map<void *, VulkanAllocationRecord> allocations_;
};

// Example usage:
//
// VulkanMemoryResource vulkanResource(device);
// std::pmr::polymorphic_allocator<std::byte> alloc(&vulkanResource);
//
// std::byte* data = alloc.allocate(1024);
// // Use the data...
// alloc.deallocate(data, 1024);
//
