#include "vma_pmr.hpp"

VulkanMemoryResource::VulkanMemoryResource(vk::Device device,
                                           vk::BufferUsageFlags buffer_usage,
                                           VmaMemoryUsage memory_usage,
                                           VmaAllocationCreateFlags flags)
    : device_(device),
      bufferUsage_(buffer_usage),
      memoryUsage_(memory_usage),
      allocationFlags_(flags) {
  SPDLOG_DEBUG(
      "VulkanMemoryResource created with usageFlags = {}, "
      "memoryUsage = {}, allocFlags = {}",
      (VkBufferUsageFlags)bufferUsage_,
      (int)memoryUsage_,
      (int)allocationFlags_);
}

VulkanMemoryResource::~VulkanMemoryResource() {
  SPDLOG_DEBUG("VulkanMemoryResource destroyed");
}

vk::Buffer VulkanMemoryResource::get_buffer_from_pointer(void *p) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(p);
  if (it == allocations_.end()) {
    // Handle unknown pointer; possibly throw an exception or return a null
    // handle
    throw std::runtime_error("Unknown pointer in get_buffer_from_pointer");
  }
  // Construct a vk::Buffer handle from the stored VkBuffer
  return vk::Buffer(it->second.buffer);
}

vk::DescriptorBufferInfo VulkanMemoryResource::make_descriptor_buffer_info(
    vk::Buffer buffer) const {
  // Get the buffer memory requirements to determine its size
  vk::BufferMemoryRequirementsInfo2 req_info{.buffer = buffer};
  vk::MemoryRequirements2 mem_reqs =
      device_.getBufferMemoryRequirements2(req_info);

  return vk::DescriptorBufferInfo{
      .buffer = buffer, .offset = 0, .range = mem_reqs.memoryRequirements.size};
}

// vk::DescriptorBufferInfo VulkanMemoryResource::get_descriptor_buffer_info(
//     void *p) {
//   std::lock_guard<std::mutex> lock(mutex_);
//   auto it = allocations_.find(p);
//   if (it == allocations_.end()) {
//     throw std::runtime_error("Unknown pointer in
//     get_descriptor_buffer_info");
//   }

//   const auto &record = it->second;
//   vk::Buffer buffer(record.buffer);
//   vk::DeviceSize size = record.allocInfo.size;

//   return vk::DescriptorBufferInfo{.buffer = buffer, .offset = 0, .range =
//   size};
// }

void *VulkanMemoryResource::do_allocate(
    std::size_t bytes, [[maybe_unused]] std::size_t alignment) {
  SPDLOG_TRACE("VulkanMemoryResource::do_allocate({}, {})", bytes, alignment);

  const vk::BufferCreateInfo buffer_create_info{
      .size = bytes,
      .usage = bufferUsage_,
  };

  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.flags = allocationFlags_;
  allocCreateInfo.usage = memoryUsage_;

  VkBuffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo allocInfo{};

  VkResult res = vmaCreateBuffer(
      g_vma_allocator,
      reinterpret_cast<const VkBufferCreateInfo *>(&buffer_create_info),
      &allocCreateInfo,
      &buffer,
      &allocation,
      &allocInfo);

  CHECK_VK_RESULT(res, "Failed to create VMA buffer");

  // The memory should already be mapped due to the
  // VMA_ALLOCATION_CREATE_MAPPED_BIT flag.
  void *mappedPtr = allocInfo.pMappedData;
  if (!mappedPtr) {
    // If it's not mapped for some reason, we can attempt to map it.
    // According to the flags, it should be mapped already, but just in case:
    res = vmaMapMemory(g_vma_allocator, allocation, &mappedPtr);
    CHECK_VK_RESULT(res, "Failed to map VMA buffer memory");
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[mappedPtr] =
        VulkanAllocationRecord{buffer, allocation, allocInfo};
  }

  return mappedPtr;
}

// Deallocate the buffer by looking up our record.
void VulkanMemoryResource::do_deallocate(
    void *p,
    [[maybe_unused]] std::size_t bytes,
    [[maybe_unused]] std::size_t alignment) {
  SPDLOG_TRACE(
      "VulkanMemoryResource::do_deallocate({}, {}, {})", p, bytes, alignment);

  VulkanAllocationRecord record;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(p);
    if (it == allocations_.end()) {
      SPDLOG_ERROR(
          "Attempted to deallocate unknown pointer from "
          "VulkanMemoryResource");
      throw std::runtime_error(
          "Unknown pointer in VulkanMemoryResource::do_deallocate");
    }
    record = it->second;
    allocations_.erase(it);
  }

  // Destroy the buffer and free the allocation
  vmaDestroyBuffer(g_vma_allocator, record.buffer, record.allocation);
}

bool VulkanMemoryResource::do_is_equal(
    const std::pmr::memory_resource &other) const noexcept {
  return dynamic_cast<const VulkanMemoryResource *>(&other) != nullptr;
}