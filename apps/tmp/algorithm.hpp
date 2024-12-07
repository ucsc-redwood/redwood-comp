#pragma once

#include <cstring>
#include <memory>

#include "vk.hpp"
#include "vma_pmr.hpp"

// #include "vulkan_resource.hpp"

// #include "buffer.hpp"

namespace vulkan {

class Algorithm final : public std::enable_shared_from_this<Algorithm> {
 public:
  explicit Algorithm(vk::Device device,
                     VulkanMemoryResource& mr,
                     const std::string_view shader_path,
                     const std::vector<vk::Buffer>& buffers);

  ~Algorithm() { destroy(); }

  void update_descriptor_sets(const std::vector<vk::Buffer>& buffers);

  template <typename T>
  std::shared_ptr<Algorithm> set_push_constants(const T data) {
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

  [[nodiscard]] std::shared_ptr<Algorithm> build();

  [[nodiscard]] bool has_push_constants() const {
    return internal_.reflected_push_constant_reported_size_ > 0;
  }

  // used by sequence class
  void record_bind_core(const vk::CommandBuffer& cmd_buf) const;
  void record_bind_push(const vk::CommandBuffer& cmd_buf) const;

  void record_dispatch(const vk::CommandBuffer& cmd_buf,
                       uint32_t data_count) const;
  void record_dispatch_blocks(const vk::CommandBuffer& cmd_buf,
                              uint32_t n_blocks) const;

 protected:
  void destroy();

 private:
  void load_and_compile_shader();
  void post_compile_reflection();

  void create_shader_module();
  void create_descriptor_set_layout();
  void create_descriptor_pool();
  void create_pipeline();

  void allocate_descriptor_sets();
  void allocate_push_constants();

  // References
  vk::Device device_ref_;
  VulkanMemoryResource& mr_ref_;

  // Core vulkan handles
  vk::ShaderModule shader_module_ = nullptr;
  vk::Pipeline pipeline_ = nullptr;
  vk::PipelineCache pipeline_cache_ = nullptr;
  vk::PipelineLayout pipeline_layout_ = nullptr;
  vk::DescriptorSetLayout descriptor_set_layout_ = nullptr;
  vk::DescriptorPool descriptor_pool_ = nullptr;
  vk::DescriptorSet descriptor_set_ = nullptr;

  std::string shader_path_;

  // Payloads
  // std::vector<std::shared_ptr<Buffer>> usm_buffers_;
  std::vector<vk::Buffer> usm_buffers_;
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
