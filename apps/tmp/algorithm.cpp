#include "algorithm.hpp"

#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan_structs.hpp>

// #include "../utils.hpp"
// #include "redwood/utils.hpp"

#include <spdlog/spdlog.h>

#include "shader_loader.hpp"
#include "spirv_reflect.h"

namespace vulkan {

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------

Algorithm::Algorithm(  // VulkanMemoryResource& mr,
                       // std::shared_ptr<VulkanMemoryResource> mr_ptr,
    VulkanMemoryResource* mr_ptr,
    const std::string_view shader_path,
    const std::vector<vk::Buffer>& buffers)
    : device_ref_(mr_ptr->get_device()),
      mr_ptr_(mr_ptr),
      shader_path_(shader_path),
      usm_buffers_(buffers) {
  SPDLOG_TRACE("Algorithm constructor");

  // load, compile, and reflect the shader.
  load_and_compile_shader();
  post_compile_reflection();
  create_shader_module();

  // create parameters (need buffers to be set),
  create_descriptor_set_layout();
  create_descriptor_pool();

  // by now, we know the descriptor layouts and push constant sizes
  allocate_descriptor_sets();
  allocate_push_constants();

  // update descriptor sets with the content from the argument passed in
  update_descriptor_sets(usm_buffers_);
}

std::shared_ptr<Algorithm> Algorithm::build() {
  SPDLOG_TRACE("Algorithm build");

  if (has_push_constants()) {
    if (push_constants_ptr_ == nullptr) {
      throw std::runtime_error(
          "Push constants detected but not allocated. Use "
          "\"->set_push_constants<T>({...})\" to allocate");
    }
  }

  create_pipeline();
  return shared_from_this();
}

// ----------------------------------------------------------------------------
// Destructor
// ----------------------------------------------------------------------------

void Algorithm::destroy() { SPDLOG_TRACE("Algorithm destroy"); }

// ----------------------------------------------------------------------------
// Compile shader to SPIRV
// ----------------------------------------------------------------------------

std::vector<uint32_t> compileShaderToSPIRV(const std::string_view filepath) {
  spdlog::debug("Compiling shader: {}", filepath);

  if (!filepath.ends_with(".comp")) {
    throw std::runtime_error("Shader source file must end with .comp");
  }

  const auto source = load_source_from_file(filepath);

  shaderc::CompileOptions options;
  // options.SetOptimizationLevel(shaderc_optimization_level_zero);
  options.SetOptimizationLevel(shaderc_optimization_level_performance);
  options.SetTargetEnvironment(shaderc_target_env_vulkan,
                               shaderc_env_version_vulkan_1_3);
  options.SetTargetSpirv(shaderc_spirv_version_1_6);

  shaderc::Compiler compiler;
  const auto result = compiler.CompileGlslToSpv(
      source, shaderc_glsl_compute_shader, filepath.data(), options);

  if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
    spdlog::error("Shader compilation failed: {}", result.GetErrorMessage());
    throw std::runtime_error(result.GetErrorMessage());
  }

  spdlog::info("Shader compiled successfully: {}", filepath);

  return {result.cbegin(), result.cend()};
}

// ----------------------------------------------------------------------------
// Reflection post-processing
// ----------------------------------------------------------------------------

void Algorithm::post_compile_reflection() {
  SPDLOG_TRACE("Algorithm post_compile_reflection");

  if (internal_.spirv_binary_.empty()) {
    throw std::runtime_error("Shader binary is empty, maybe not compiled?");
  }

  SpvReflectShaderModule reflected_module;

  if (spvReflectCreateShaderModule(
          internal_.spirv_binary_.size() * sizeof(uint32_t),
          internal_.spirv_binary_.data(),
          &reflected_module) != SPV_REFLECT_RESULT_SUCCESS) {
    throw std::runtime_error("Failed to reflect SPIR-V shader module");
  }

  // Step 1. Get the entry point
  auto entry = spvReflectGetEntryPoint(&reflected_module, "main");
  if (!entry) {
    throw std::runtime_error("Failed to get entry point");
  }

  // Step 2. Get the work group size
  internal_.reflected_workgroup_size_[0] = entry->local_size.x;
  internal_.reflected_workgroup_size_[1] = entry->local_size.y;
  internal_.reflected_workgroup_size_[2] = entry->local_size.z;

  spdlog::debug("Workgroup size: {} {} {}",
                internal_.reflected_workgroup_size_[0],
                internal_.reflected_workgroup_size_[1],
                internal_.reflected_workgroup_size_[2]);

  // Step 3. Get the push constant size
  uint32_t push_constant_count = 0;

  if (spvReflectEnumeratePushConstantBlocks(
          &reflected_module, &push_constant_count, nullptr) !=
      SPV_REFLECT_RESULT_SUCCESS) {
    throw std::runtime_error("Failed to get push constant count");
  }

  spdlog::debug("Push constant count: {}", push_constant_count);

  // in my case, there should be either 0 or 1 push constants
  assert(push_constant_count == 1 || push_constant_count == 0);

  // should be only one in my case
  std::vector<SpvReflectBlockVariable*> push_constant_blocks(
      push_constant_count);

  if (spvReflectEnumeratePushConstantBlocks(&reflected_module,
                                            &push_constant_count,
                                            push_constant_blocks.data()) !=
      SPV_REFLECT_RESULT_SUCCESS) {
    throw std::runtime_error("Failed to get push constant blocks");
  }

  spdlog::debug("Push Constants:");
  for (const auto* push_constant : push_constant_blocks) {
    spdlog::debug("  Name: {}",
                  push_constant->name ? push_constant->name : "[Unnamed]");
    spdlog::debug("  Size: {} bytes", push_constant->size);
    spdlog::debug("  Offset: {}", push_constant->offset);
    spdlog::debug("  Members:");

    for (uint32_t i = 0; i < push_constant->member_count; ++i) {
      const auto& member = push_constant->members[i];

      internal_.reflected_push_constant_accumulated_size_ += member.size;

      spdlog::debug("    {} (Offset: {}, Size: {})",
                    member.name ? member.name : "[Unnamed]",
                    member.offset,
                    member.size);
    }

    internal_.reflected_push_constant_reported_size_ = push_constant->size;
  }

  spdlog::debug("Push constant reported size: {}",
                internal_.reflected_push_constant_reported_size_);
  spdlog::debug("Push constant accumulated size: {}",
                internal_.reflected_push_constant_accumulated_size_);

  // Step 4. Get the descriptor bindings

  spvReflectDestroyShaderModule(&reflected_module);
}

// ----------------------------------------------------------------------------
// Load and compile shader
// ----------------------------------------------------------------------------

void Algorithm::load_and_compile_shader() {
  SPDLOG_TRACE("Algorithm load_and_compile_shader");

  if (shader_path_.empty()) {
    throw std::runtime_error("Shader path is empty");
  }

  internal_.spirv_binary_ = compileShaderToSPIRV(shader_path_);
}

// ----------------------------------------------------------------------------
// Create shader module
// ----------------------------------------------------------------------------

void Algorithm::create_shader_module() {
  SPDLOG_TRACE("Algorithm create_shader_module");

  if (internal_.spirv_binary_.empty()) {
    throw std::runtime_error("Shader binary is empty");
  }

  const vk::ShaderModuleCreateInfo create_info{
      .codeSize = internal_.spirv_binary_.size() * sizeof(uint32_t),
      .pCode = internal_.spirv_binary_.data(),
  };

  shader_module_ = device_ref_.createShaderModule(create_info);

  spdlog::debug("Shader module [{}] created successfully", shader_path_);
}

// ----------------------------------------------------------------------------
// Descriptor
// ----------------------------------------------------------------------------

void Algorithm::create_descriptor_pool() {
  SPDLOG_TRACE("Algorithm create_descriptor_pool");

  const std::vector pool_sizes{
      vk::DescriptorPoolSize{
          .type = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = static_cast<uint32_t>(usm_buffers_.size()),
      },
  };

  const vk::DescriptorPoolCreateInfo create_info{
      .maxSets = 1,
      .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
      .pPoolSizes = pool_sizes.data(),
  };

  descriptor_pool_ = device_ref_.createDescriptorPool(create_info);
}

// ----------------------------------------------------------------------------
// Descriptor set layout
// ----------------------------------------------------------------------------

void Algorithm::create_descriptor_set_layout() {
  SPDLOG_TRACE("Algorithm create_descriptor_set_layout");

  std::vector<vk::DescriptorSetLayoutBinding> bindings;
  bindings.reserve(usm_buffers_.size());

  for (uint32_t i = 0; i < usm_buffers_.size(); ++i) {
    bindings.emplace_back(vk::DescriptorSetLayoutBinding{
        .binding = i,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
    });
  }

  const vk::DescriptorSetLayoutCreateInfo create_info{
      .bindingCount = static_cast<uint32_t>(bindings.size()),
      .pBindings = bindings.data(),
  };

  descriptor_set_layout_ = device_ref_.createDescriptorSetLayout(create_info);
}

// ----------------------------------------------------------------------------
// Allocate descriptor sets
// ----------------------------------------------------------------------------

void Algorithm::allocate_descriptor_sets() {
  SPDLOG_TRACE("Algorithm allocate_descriptor_sets");

  if (descriptor_pool_ == nullptr || descriptor_set_layout_ == nullptr) {
    throw std::runtime_error(
        "Descriptor pool or set layout is not initialized");
  }

  const vk::DescriptorSetAllocateInfo allocate_info{
      .descriptorPool = descriptor_pool_,
      .descriptorSetCount = 1,
      .pSetLayouts = &descriptor_set_layout_,
  };

  descriptor_set_ = device_ref_.allocateDescriptorSets(allocate_info).front();
}

// ----------------------------------------------------------------------------
// Pipeline
// ----------------------------------------------------------------------------

void Algorithm::create_pipeline() {
  SPDLOG_TRACE("Algorithm create_pipeline");

  if (descriptor_set_layout_ == nullptr) {
    throw std::runtime_error("Descriptor set layout is not initialized");
  }

  std::vector<vk::PushConstantRange> push_constant_ranges;

  if (has_push_constants()) {
    push_constant_ranges.emplace_back(vk::PushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = internal_.reflected_push_constant_reported_size_,
    });
  }

  const vk::PipelineLayoutCreateInfo pipeline_layout_create_info{
      .setLayoutCount = 1,
      .pSetLayouts = &descriptor_set_layout_,
      .pushConstantRangeCount =
          static_cast<uint32_t>(push_constant_ranges.size()),
      .pPushConstantRanges =
          push_constant_ranges.empty() ? nullptr : push_constant_ranges.data()};

  pipeline_layout_ =
      device_ref_.createPipelineLayout(pipeline_layout_create_info);

  pipeline_cache_ =
      device_ref_.createPipelineCache(vk::PipelineCacheCreateInfo{});

  const vk::PipelineShaderStageCreateInfo shader_stage_create_info{
      .stage = vk::ShaderStageFlagBits::eCompute,
      .module = shader_module_,
      .pName = "main",
  };

  const vk::ComputePipelineCreateInfo pipeline_create_info{
      .stage = shader_stage_create_info,
      .layout = pipeline_layout_,
      .basePipelineHandle = nullptr,
  };

  pipeline_ =
      device_ref_.createComputePipeline(pipeline_cache_, pipeline_create_info)
          .value;

  spdlog::debug("Pipeline [{}] created successfully", shader_path_);
}

// ----------------------------------------------------------------------------
// Update descriptor sets
// ----------------------------------------------------------------------------

void Algorithm::update_descriptor_sets(const std::vector<vk::Buffer>& buffers) {
  SPDLOG_TRACE("Algorithm update_descriptor_sets");

  if (descriptor_set_ == nullptr) {
    throw std::runtime_error("Descriptor set is not initialized");
  }

  std::vector<vk::WriteDescriptorSet> compute_write_descriptor_sets;
  compute_write_descriptor_sets.reserve(buffers.size());

  std::vector<vk::DescriptorBufferInfo> buffer_infos;
  buffer_infos.reserve(buffers.size());

  for (uint32_t i = 0; i < buffers.size(); ++i) {
    // buffer_infos.emplace_back(buffers[i].get_descriptor_buffer_info());

    // buffer_infos.emplace_back(mr_ref_.make_descriptor_buffer_info(buffers[i]));
    buffer_infos.emplace_back(mr_ptr_->make_descriptor_buffer_info(buffers[i]));

    compute_write_descriptor_sets.emplace_back(vk::WriteDescriptorSet{
        .dstSet = descriptor_set_,
        .dstBinding = i,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pBufferInfo = &buffer_infos.back(),
    });
  }

  device_ref_.updateDescriptorSets(
      static_cast<uint32_t>(compute_write_descriptor_sets.size()),
      compute_write_descriptor_sets.data(),
      0,
      nullptr);
}

// ----------------------------------------------------------------------------
// Allocate push constants
// ----------------------------------------------------------------------------

void Algorithm::allocate_push_constants() {
  SPDLOG_TRACE("Algorithm allocate_push_constants");

  if (push_constants_ptr_) {
    spdlog::warn("Push constants already allocated");
    return;
  }

  push_constants_ptr_ = std::make_unique<std::byte[]>(
      internal_.reflected_push_constant_accumulated_size_);
}

// ----------------------------------------------------------------------------
// record_bind_core
// ----------------------------------------------------------------------------

void Algorithm::record_bind_core(const vk::CommandBuffer& cmd_buf) const {
  SPDLOG_TRACE("Algorithm record_bind_core");

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
  cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             pipeline_layout_,
                             0,
                             descriptor_set_,
                             nullptr);
}

// ----------------------------------------------------------------------------
// record_bind_push
// ----------------------------------------------------------------------------

void Algorithm::record_bind_push(const vk::CommandBuffer& cmd_buf) const {
  SPDLOG_TRACE("Algorithm record_bind_push");

  spdlog::debug("Pushing constants of size {}", get_push_constants_size());

  if (push_constants_ptr_ == nullptr) {
    throw std::runtime_error("Push constants not allocated");
  }

  cmd_buf.pushConstants(pipeline_layout_,
                        vk::ShaderStageFlagBits::eCompute,
                        0,
                        get_push_constants_size(),
                        push_constants_ptr_.get());
}

// ----------------------------------------------------------------------------
// record_dispatch_with_blocks
// ----------------------------------------------------------------------------

// void Algorithm::record_dispatch_with_blocks(const vk::CommandBuffer& cmd_buf,
//                                             const uint32_t n_blocks) {
//   SPD_TRACE_FUNC

//   cmd_buf.dispatch(n_blocks, 1, 1);
// }

// void Algorithm::record_dispatch(const vk::CommandBuffer& cmd_buf) const {
//   SPD_TRACE_FUNC

//   spdlog::debug("Dispatching 1 block of size {} threads per block",
//                 get_workgroup_size_x());

//   cmd_buf.dispatch(1, 1, 1);
// }

void Algorithm::record_dispatch(const vk::CommandBuffer& cmd_buf,
                                const uint32_t data_count) const {
  SPDLOG_TRACE("Algorithm record_dispatch");

  const auto workgroup_size_x = get_workgroup_size_x();
  const auto n_blocks = (data_count + workgroup_size_x - 1) / workgroup_size_x;

  spdlog::debug(
      "Dispatching {} blocks of size {} threads per block on #{} "
      "data",
      n_blocks,
      workgroup_size_x,
      data_count);

  cmd_buf.dispatch(n_blocks, 1, 1);
}

void Algorithm::record_dispatch_blocks(const vk::CommandBuffer& cmd_buf,
                                       const uint32_t n_blocks) const {
  SPDLOG_TRACE("Algorithm record_dispatch_blocks");

  spdlog::debug("Dispatching {} blocks of size {} threads per block",
                n_blocks,
                get_workgroup_size_x());

  cmd_buf.dispatch(n_blocks, 1, 1);
}

}  // namespace vulkan
