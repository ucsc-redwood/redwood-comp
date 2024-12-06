#include "buffer.hpp"

#include "../utils.hpp"
#include "base_engine.hpp"

namespace vulkan {

Buffer::Buffer(std::shared_ptr<vk::Device> device_ptr,
               const vk::DeviceSize size,
               const vk::BufferUsageFlags buffer_usage,
               const VmaMemoryUsage memory_usage,
               const VmaAllocationCreateFlags flags)
    : VulkanResource(std::move(device_ptr)), size_(size) {
  SPD_TRACE_FUNC

  const vk::BufferCreateInfo buffer_create_info{
      .size = size,
      .usage = buffer_usage,
  };

  const VmaAllocationCreateInfo memory_info{
      .flags = flags,
      .usage = memory_usage,
      .requiredFlags = 0,
      .preferredFlags = 0,
      .memoryTypeBits = 0,
      .pool = VK_NULL_HANDLE,
      .pUserData = nullptr,
      .priority = 0.0f,
  };

  VmaAllocationInfo allocation_info{};

  vmaCreateBuffer(
      g_vma_allocator,
      reinterpret_cast<const VkBufferCreateInfo *>(&buffer_create_info),
      &memory_info,
      reinterpret_cast<VkBuffer *>(&this->get_handle()),
      &allocation_,
      &allocation_info);

  spdlog::debug("Created buffer [{}]", static_cast<void *>(this));
  spdlog::debug("\tsize: {}", allocation_info.size);
  spdlog::debug("\toffset: {}", allocation_info.offset);
  spdlog::debug("\tmemoryType: {}", allocation_info.memoryType);
  spdlog::debug("\tmappedData: {}", allocation_info.pMappedData);

  memory_ = static_cast<vk::DeviceMemory>(allocation_info.deviceMemory);
  mapped_data_ = static_cast<std::byte *>(allocation_info.pMappedData);
}

void Buffer::destroy() {
  SPD_TRACE_FUNC

  if (allocation_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(g_vma_allocator, this->get_handle(), allocation_);
  }
}

}  // namespace vulkan
