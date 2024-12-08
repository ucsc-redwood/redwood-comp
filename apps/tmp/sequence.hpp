#pragma once

#include "algorithm.hpp"

namespace vulkan {

/**
 * @brief Sequence handles command recording and execution for compute
 * operations
 *
 * The Sequence class manages Vulkan command buffers and synchronization for
 * compute shader execution. It provides:
 * - Command buffer management
 * - Command recording utilities
 * - Asynchronous kernel launch
 * - Synchronization primitives
 *
 * Example usage:
 * ```cpp
 * // Create sequence
 * auto sequence = std::make_shared<Sequence>(device, compute_queue,
 * queue_index);
 *
 * // Record commands
 * sequence->record_commands_with_blocks(algorithm.get(), work_groups);
 *
 * // Execute asynchronously
 * sequence->launch_kernel_async();
 *
 * // Do other work...
 *
 * // Wait for completion
 * sequence->sync();
 * ```
 */
class Sequence {
 public:
  explicit Sequence(vk::Device device_ref,
                    vk::Queue compute_queue_ref,
                    uint32_t compute_queue_index);

  ~Sequence() { destroy(); }

  void cmd_begin() const;
  void cmd_end() const;

  void record_commands(const Algorithm* algo, uint32_t data_count) const;
  void record_commands_with_blocks(const Algorithm* algo,
                                   uint32_t n_blocks) const;

  void launch_kernel_async();
  void sync() const;

 protected:
  void destroy();

 private:
  void create_sync_objects();
  void create_command_pool();
  void create_command_buffer();

  vk::Device device_ref_;
  vk::Queue compute_queue_ref_;

  // std::shared_ptr<vk::Queue> compute_queue_ptr_;
  uint32_t compute_queue_index_;

  vk::CommandBuffer handle_;
  vk::CommandPool command_pool_;
  vk::Fence fence_;
};

}  // namespace vulkan
