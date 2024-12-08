#pragma once

#include <cstring>
#include <memory>

#include "buffer.hpp"

namespace vulkan {

/**
 * @brief A compute shader pipeline wrapper for Vulkan compute operations
 *
 * The Algorithm class encapsulates a complete compute pipeline including:
 * - Shader module loading and management
 * - Pipeline and descriptor set layout creation
 * - Buffer binding and descriptor set updates
 * - Push constant management
 * - Command recording utilities
 *
 * The class uses a builder pattern for configuration, allowing method chaining
 * that ends with build() to create the pipeline.
 *
 * Example usage:
 * @code
 * // Create algorithm with buffers
 * auto algo = std::make_shared<Algorithm>(
 *     device,
 *     "shaders/compute.spv",
 *     BufferVec{input_buf, output_buf}
 * );
 *
 * // Configure push constants and build
 * struct PushConstants {
 *     float scale;
 *     int count;
 * } pc{1.5f, 1024};
 *
 * algo->set_push_constants(pc)
 *     ->build();
 *
 * // Use in a sequence
 * sequence->record(algo);
 * @endcode
 */
class Algorithm final : public VulkanResource<vk::ShaderModule>,
                        public std::enable_shared_from_this<Algorithm> {
 public:
  /**
   * @brief Constructs an Algorithm instance
   *
   * Creates a new compute pipeline from the specified shader file and
   * configures it with the provided buffer bindings.
   *
   * @param device_ptr Vulkan logical device
   * @param shader_path Path to the SPIR-V compute shader file
   * @param buffers Vector of buffers to bind to the shader's descriptor sets
   *
   * @note The shader must be a valid SPIR-V compute shader
   * @note Buffer count and types must match the shader's descriptor bindings
   */
  explicit Algorithm(std::shared_ptr<vk::Device> device_ptr,
                     const std::string_view shader_path,
                     const BufferVec& buffers);

  ~Algorithm() override { destroy(); }

  /**
   * @brief Updates descriptor sets with new buffer bindings
   *
   * Allows changing the buffers bound to the shader without recreating the
   * pipeline. The number and types of buffers must match the original
   * configuration.
   *
   * @param buffers New vector of buffers to bind
   * @throws std::runtime_error if buffer count/types don't match shader
   * requirements
   */
  void update_descriptor_sets(const BufferVec& buffers);

  /**
   * @brief Sets push constant data for the shader
   *
   * Updates the push constant data that will be passed to the shader during
   * execution. The data type and size must match the push constant block
   * defined in the shader.
   *
   * @tparam T Type of the push constant data structure
   * @param data Push constant data to set
   * @return shared_ptr to this Algorithm for method chaining
   * @throws std::runtime_error if shader doesn't declare push constants or if
   * size mismatch
   *
   * Example:
   * @code
   * struct PushData {
   *     float scale;
   *     int count;
   * } data{1.0f, 100};
   *
   * algorithm->set_push_constants(data);
   * @endcode
   */
  template <typename T>
  std::shared_ptr<Algorithm> set_push_constants(const T data) {
    // spdlog::debug(
    //     "Setting push constants - reported_size = {}, accumulated_size = {}",
    //     internal_.reflected_push_constant_reported_size_,
    //     internal_.reflected_push_constant_accumulated_size_);

    if (!has_push_constants()) {
      throw std::runtime_error(
          "Push constants not reported by shader, thus don't set push "
          "constants");
    }

    if (push_constants_ptr_ == nullptr) {
      // spdlog::warn("Push constants not allocated, allocating");

      allocate_push_constants();
    }

    update_push_constants(data);
    return shared_from_this();
  }

  /**
   * @brief Updates existing push constant data
   *
   * @tparam T Type of the push constant data structure
   * @param data New push constant data
   * @throws std::runtime_error if push constants haven't been allocated
   */
  template <typename T>
  void update_push_constants(const T data) {
    if (push_constants_ptr_ == nullptr) {
      throw std::runtime_error("Push constants not allocated");
    }

    assert(this->get_push_constants_size() == sizeof(data));

    std::memcpy(push_constants_ptr_.get(),
                std::bit_cast<const std::byte*>(&data),
                get_push_constants_size());
  }

  /**
   * @brief Builds the compute pipeline
   *
   * Finalizes the pipeline configuration and creates all necessary Vulkan
   * objects. Must be called after configuration and before using the algorithm.
   *
   * @return shared_ptr to this Algorithm for method chaining
   * @throws std::runtime_error if pipeline creation fails
   */
  [[nodiscard]] std::shared_ptr<Algorithm> build();

  /**
   * @brief Checks if the shader uses push constants
   *
   * @return true if the shader declares push constants, false otherwise
   */
  [[nodiscard]] bool has_push_constants() const {
    return internal_.reflected_push_constant_reported_size_ > 0;
  }

  // used by sequence class
  void record_bind_core(const vk::CommandBuffer& cmd_buf) const;
  void record_bind_push(const vk::CommandBuffer& cmd_buf) const;
  // void record_dispatch(const vk::CommandBuffer& cmd_buf) const;
  void record_dispatch(const vk::CommandBuffer& cmd_buf,
                       uint32_t data_count) const;
  void record_dispatch_blocks(const vk::CommandBuffer& cmd_buf,
                              uint32_t n_blocks) const;

 protected:
  void destroy() override;

 private:
  void load_and_compile_shader();
  void post_compile_reflection();

  void create_shader_module();
  void create_descriptor_set_layout();
  void create_descriptor_pool();
  void create_pipeline();

  void allocate_descriptor_sets();
  void allocate_push_constants();

  // Core vulkan handles
  vk::Pipeline pipeline_ = nullptr;
  vk::PipelineCache pipeline_cache_ = nullptr;
  vk::PipelineLayout pipeline_layout_ = nullptr;
  vk::DescriptorSetLayout descriptor_set_layout_ = nullptr;
  vk::DescriptorPool descriptor_pool_ = nullptr;
  vk::DescriptorSet descriptor_set_ = nullptr;

  std::string shader_path_;

  // Payloads
  std::vector<std::shared_ptr<Buffer>> usm_buffers_;
  std::unique_ptr<std::byte[]> push_constants_ptr_ = nullptr;

  [[nodiscard]] uint32_t get_push_constants_size() const {
    return internal_.reflected_push_constant_accumulated_size_;
  }

  [[nodiscard]] uint32_t get_workgroup_size_x() const {
    return internal_.reflected_workgroup_size_[0];
  }

  // internal use, intermediate data (e.g., compiled shader, reflection data)
  struct {
    std::vector<uint32_t> spirv_binary_;

    uint32_t reflected_workgroup_size_[3] = {0, 0, 0};
    uint32_t reflected_push_constant_reported_size_ = 0;
    uint32_t reflected_push_constant_accumulated_size_ = 0;
  } internal_;
};

}  // namespace vulkan
