#include "sequence.hpp"

#include "../utils.hpp"

namespace vulkan {

Sequence::Sequence(std::shared_ptr<vk::Device> device_ptr,
                   std::shared_ptr<vk::Queue> compute_queue_ptr,
                   uint32_t compute_queue_index)
    : VulkanResource<vk::CommandBuffer>(device_ptr),
      compute_queue_ptr_(std::move(compute_queue_ptr)),
      compute_queue_index_(compute_queue_index) {
  SPD_TRACE_FUNC

  create_sync_objects();
  create_command_pool();
  create_command_buffer();
}

void Sequence::destroy() { SPD_TRACE_FUNC }

void Sequence::create_command_pool() {
  SPD_TRACE_FUNC

  const vk::CommandPoolCreateInfo create_info{
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = compute_queue_index_,
  };

  command_pool_ = device_ptr_->createCommandPool(create_info);
}

void Sequence::create_sync_objects() {
  SPD_TRACE_FUNC

  constexpr vk::FenceCreateInfo create_info{};
  fence_ = device_ptr_->createFence(create_info);
}

void Sequence::create_command_buffer() {
  SPD_TRACE_FUNC

  const vk::CommandBufferAllocateInfo allocate_info{
      .commandPool = command_pool_,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1,
  };

  handle_ = device_ptr_->allocateCommandBuffers(allocate_info).front();
}

void Sequence::cmd_begin() const {
  SPD_TRACE_FUNC

  constexpr vk::CommandBufferBeginInfo begin_info{
      .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
  };

  handle_.begin(begin_info);
}

void Sequence::cmd_end() const {
  SPD_TRACE_FUNC

  handle_.end();
}

void Sequence::launch_kernel_async() {
  SPD_TRACE_FUNC

  vk::SubmitInfo submit_info{
      .commandBufferCount = 1,
      .pCommandBuffers = &handle_,
  };

  compute_queue_ptr_->submit(submit_info, fence_);
}

void Sequence::sync() const {
  SPD_TRACE_FUNC

  if (device_ptr_->waitForFences(1, &fence_, true, UINT64_MAX) !=
      vk::Result::eSuccess) {
    throw std::runtime_error("Failed to sync sequence");
  }

  if (device_ptr_->resetFences(1, &fence_) != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to reset sequence");
  }
}

// void Sequence::record_commands(const Algorithm* algo) const {
//   cmd_begin();

//   algo->record_bind_core(handle_);

//   if (algo->has_push_constants()) {
//     algo->record_bind_push(handle_);
//   }

//   algo->record_dispatch(handle_);

//   cmd_end();
// }

void Sequence::record_commands(const Algorithm* algo,
                               const uint32_t data_count) const {
  cmd_begin();

  algo->record_bind_core(handle_);

  if (algo->has_push_constants()) {
    algo->record_bind_push(handle_);
  }

  algo->record_dispatch(handle_, data_count);

  cmd_end();
}

void Sequence::record_commands_with_blocks(const Algorithm* algo,
                                           const uint32_t n_blocks) const {
  cmd_begin();

  algo->record_bind_core(handle_);
  if (algo->has_push_constants()) {
    algo->record_bind_push(handle_);
  }

  algo->record_dispatch_blocks(handle_, n_blocks);

  cmd_end();
}

}  // namespace vulkan
