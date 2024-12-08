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
class Sequence final : public VulkanResource<vk::CommandBuffer> {
 public:
  /**
   * @brief Constructs a Sequence instance
   *
   * @param device_ptr Vulkan logical device
   * @param compute_queue_ptr Queue to submit commands to
   * @param compute_queue_index Index of the compute queue family
   */
  explicit Sequence(std::shared_ptr<vk::Device> device_ptr,
                    std::shared_ptr<vk::Queue> compute_queue_ptr,
                    uint32_t compute_queue_index);

  ~Sequence() override { Sequence::destroy(); }

  /**
   * @brief Begin recording commands
   * Used for custom command recording scenarios
   */
  void cmd_begin() const;

  /**
   * @brief End command recording
   * Used for custom command recording scenarios
   */
  void cmd_end() const;

  // basically all in, automatically with blocks
  // void record_commands(const Algorithm* algo) const;
  void record_commands(const Algorithm* algo, uint32_t data_count) const;
  void record_commands_with_blocks(const Algorithm* algo,
                                   uint32_t n_blocks) const;

  /**
   * @brief Submit commands to queue asynchronously
   *
   * The commands will begin executing but this call doesn't wait for
   * completion. Call sync() to wait for the work to finish.
   */
  void launch_kernel_async();

  /**
   * @brief Wait for submitted work to complete
   *
   * Blocks until all previously submitted commands have finished executing.
   */
  void sync() const;

 protected:
  void destroy() override;

 private:
  void create_sync_objects();
  void create_command_pool();
  void create_command_buffer();

  std::shared_ptr<vk::Queue> compute_queue_ptr_;
  uint32_t compute_queue_index_;

  vk::CommandPool command_pool_;
  vk::Fence fence_;
};

}  // namespace vulkan
