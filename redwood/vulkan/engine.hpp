#pragma once

// #include "algorithm.hpp"
#include "base_engine.hpp"
#include "buffer.hpp"
// #include "sequence.hpp"
// #include "typed_buffer.hpp"
// #include "utils.hpp"

/**
 * @brief A high-level Vulkan compute engine that manages compute resources
 *
 * The Engine class provides a simplified interface for:
 * - Creating and managing buffers (both typed and untyped)
 * - Creating compute algorithms from shader files
 * - Managing compute sequences for execution
 *
 * When created with manage_resources=true (default), the engine will
 * automatically track and manage the lifetime of created resources. Otherwise,
 * resources must be manually managed by the caller.
 *
 * Example usage:
 * @code
 * // Create engine with automatic resource management
 * Engine engine;
 *
 * // Create buffers
 * auto input_buffer = engine.typed_buffer<float>(1024); // buffer for 1024
 * floats auto output_buffer = engine.typed_buffer<float>(1024);
 *
 * // Create compute algorithm from shader
 * auto algorithm = engine.algorithm("shaders/compute.spv", {input_buffer,
 * output_buffer});
 *
 * // Create and execute compute sequence
 * auto sequence = engine.sequence();
 * sequence->record(algorithm)
 *         ->submit()
 *         ->wait();
 * @endcode
 */
namespace vulkan {

class Engine final : public BaseEngine {
 public:
  /**
   * @brief Constructs an Engine instance
   *
   * @param manage_resources If true (default), the engine will automatically
   * manage the lifetime of created resources (buffers, algorithms, sequences).
   *                        If false, resources must be manually managed.
   */
  explicit Engine(const bool enable_validation_layer = true,
                  const bool manage_resources = true)
      : BaseEngine(enable_validation_layer),
        manage_resources_{manage_resources} {
    // SPD_TRACE_FUNC
  }

  ~Engine() = default;

  /**
   * @brief Creates an untyped Vulkan buffer
   *
   * Creates a new Buffer instance with the specified parameters. If resource
   * management is enabled, the buffer's lifetime will be tracked by the engine.
   *
   * @tparam Args Constructor argument types for Buffer
   * @param args Arguments forwarded to Buffer constructor
   * @return std::shared_ptr<Buffer> Shared pointer to the created buffer
   *
   * @note Prefer typed_buffer<T> when working with known data types
   *
   * Example:
   * @code
   * auto buffer = engine.buffer(
   *     1024,  // size in bytes
   *     vk::BufferUsageFlagBits::eStorageBuffer,
   *     VMA_MEMORY_USAGE_AUTO
   * );
   * @endcode
   */
  template <typename... Args>
    requires std::
        is_constructible_v<Buffer, std::shared_ptr<vk::Device>, Args...>
      [[nodiscard]] std::shared_ptr<Buffer> buffer(Args&&... args) {
    // SPD_TRACE_FUNC

    const auto buffer =
        std::make_shared<Buffer>(this->get_device_ptr(), args...);

    if (manage_resources_) {
      buffers_.push_back(buffer);
    }

    return buffer;
  }

  // /**
  //  * @brief Creates a typed Vulkan buffer
  //  *
  //  * Creates a new TypedBuffer instance for a specific data type. Provides
  //  * type-safe access to buffer memory. If resource management is enabled, the
  //  * buffer's lifetime will be tracked by the engine.
  //  *
  //  * @tparam T Data type for the buffer (e.g., float, int, custom struct)
  //  * @tparam Args Constructor argument types for TypedBuffer
  //  * @param args Arguments forwarded to TypedBuffer constructor
  //  * @return std::shared_ptr<TypedBuffer<T>> Shared pointer to the created typed
  //  * buffer
  //  *
  //  * Example:
  //  * @code
  //  * // Create a buffer for 1024 floats
  //  * auto float_buffer = engine.typed_buffer<float>(
  //  *     1024,  // count of elements
  //  *     vk::BufferUsageFlagBits::eStorageBuffer,
  //  *     VMA_MEMORY_USAGE_AUTO
  //  * );
  //  * @endcode
  //  */
  // template <typename T, typename... Args>
  //   requires std::is_constructible_v<TypedBuffer<T>,
  //                                    std::shared_ptr<vk::Device>,
  //                                    Args...>
  // [[nodiscard]] auto typed_buffer(Args&&... args)
  //     -> std::shared_ptr<TypedBuffer<T>> {
  //   // SPD_TRACE_FUNC

  //   auto buf =
  //       std::make_shared<TypedBuffer<T>>(this->get_device_ptr(), args...);

  //   if (manage_resources_) {
  //     buffers_.push_back(buf);
  //   }

  //   return buf;
  // }

  // /**
  //  * @brief Creates a compute algorithm from a shader
  //  *
  //  * Creates a new Algorithm instance using the specified shader and buffer
  //  * bindings. If resource management is enabled, the algorithm's lifetime will
  //  * be tracked by the engine.
  //  *
  //  * @param shader_path Path to the SPIR-V compute shader file
  //  * @param buffers Vector of buffers to bind to the shader
  //  * @return std::shared_ptr<Algorithm> Shared pointer to the created algorithm
  //  *
  //  * Example:
  //  * @code
  //  * auto algorithm = engine.algorithm(
  //  *     "shaders/compute.spv",
  //  *     {input_buffer, output_buffer}
  //  * );
  //  * @endcode
  //  */
  // [[nodiscard]] std::shared_ptr<Algorithm> algorithm(
  //     const std::string_view shader_path, const BufferVec& buffers) {
  //   // SPD_TRACE_FUNC

  //   const auto algorithm_ptr = std::make_shared<Algorithm>(
  //       this->get_device_ptr(), shader_path, buffers);

  //   if (manage_resources_) {
  //     algorithms_.push_back(algorithm_ptr);
  //   }

  //   return algorithm_ptr;
  // }

  // /**
  //  * @brief Creates a command sequence for executing compute operations
  //  *
  //  * Creates a new Sequence instance for recording and executing compute
  //  * commands. If resource management is enabled, the sequence's lifetime will
  //  * be tracked by the engine.
  //  *
  //  * @return std::shared_ptr<Sequence> Shared pointer to the created sequence
  //  *
  //  * Example:
  //  * @code
  //  * auto sequence = engine.sequence();
  //  * sequence->record(algorithm)
  //  *         ->submit()
  //  *         ->wait();
  //  * @endcode
  //  */
  // [[nodiscard]] std::shared_ptr<Sequence> sequence() {
  //   // SPD_TRACE_FUNC

  //   const auto sequence_ptr =
  //       std::make_shared<Sequence>(this->get_device_ptr(),
  //                                  this->get_compute_queue_ptr(),
  //                                  this->get_compute_queue_family_index());

  //   if (manage_resources_) {
  //     sequences_.push_back(sequence_ptr);
  //   }

  //   return sequence_ptr;
  // }

  // --------------------------------------------------------------------------
  // Getters
  // --------------------------------------------------------------------------

  // read-only access to buffers
  [[nodiscard]] auto get_buffers() const
      -> const std::vector<std::weak_ptr<Buffer>>& {
    return buffers_;
  }

  // get total memory usage in bytes
  [[nodiscard]] auto get_total_memory_usage() const -> size_t {
    size_t mem = 0;
    for (const auto& buf : buffers_) {
      mem += buf.lock()->get_size_in_bytes();
    }
    return mem;
  }

 private:
  std::vector<std::weak_ptr<Buffer>> buffers_;
  // std::vector<std::weak_ptr<Algorithm>> algorithms_;
  // std::vector<std::weak_ptr<Sequence>> sequences_;

  bool manage_resources_ = true;
};

}  // namespace vulkan
