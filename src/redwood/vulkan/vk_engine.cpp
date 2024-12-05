#include "vk_engine.hpp"

#include "../utils.hpp"

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

// ----------------------------------------------------------------------------
// Global variables
// ----------------------------------------------------------------------------

VmaAllocator g_vma_allocator = nullptr;

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------

VulkanEngine::VulkanEngine(const bool enable_validation_layer) {
  SPD_TRACE_FUNC

  initialize_dynamic_loader();

  if (enable_validation_layer) {
    request_validation_layer();
  }

  create_instance();
  create_physical_device();
  create_device();

  initialize_vma_allocator();
}

// ----------------------------------------------------------------------------
// Destructor
// ----------------------------------------------------------------------------

void VulkanEngine::destroy() {
  SPD_TRACE_FUNC

  if (g_vma_allocator) {
    vmaDestroyAllocator(g_vma_allocator);
  }
}

// ----------------------------------------------------------------------------
// Dynamic loader
// ----------------------------------------------------------------------------

void VulkanEngine::initialize_dynamic_loader() {
  SPD_TRACE_FUNC

  dl_ = vk::DynamicLoader();

  vkGetInstanceProcAddr_ =
      dl_.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");

  if (!vkGetInstanceProcAddr_) {
    throw std::runtime_error("vkGetInstanceProcAddr not found");
  }

  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr_);

  vkGetDeviceProcAddr_ =
      dl_.getProcAddress<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");

  if (!vkGetDeviceProcAddr_) {
    throw std::runtime_error("vkGetDeviceProcAddr not found");
  }
}

// ----------------------------------------------------------------------------
// Validation layer
// ----------------------------------------------------------------------------

void VulkanEngine::request_validation_layer() {
  SPD_TRACE_FUNC

  constexpr auto validationLayerName = "VK_LAYER_KHRONOS_validation";

  const auto availableLayers = vk::enumerateInstanceLayerProperties();
  bool layerFound = std::ranges::any_of(
      availableLayers, [validationLayerName](const auto &layer) {
        return std::strcmp(layer.layerName.data(), validationLayerName) == 0;
      });

  if (!layerFound) {
    spdlog::warn(
        "Validation layer requested but not available, continuing without "
        "it...");
    return;
  }

  enabledLayers_.push_back(validationLayerName);
}

// ----------------------------------------------------------------------------
// Instance
// ----------------------------------------------------------------------------

void VulkanEngine::create_instance() {
  SPD_TRACE_FUNC

  constexpr vk::ApplicationInfo appInfo{
      .pApplicationName = "Vulkan Compute Example",
      .applicationVersion = 1,
      .pEngineName = "No Engine",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_3,
  };

  const vk::InstanceCreateInfo instanceCreateInfo{
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(enabledLayers_.size()),
      .ppEnabledLayerNames = enabledLayers_.data(),
  };

  instance_ = vk::createInstance(instanceCreateInfo);

  VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);
}

// ----------------------------------------------------------------------------
// Physical device
// ----------------------------------------------------------------------------

void VulkanEngine::create_physical_device(vk::PhysicalDeviceType type) {
  SPD_TRACE_FUNC

  if (!instance_) {
    throw std::runtime_error("Instance is not valid");
  }

  // Get all physical devices
  const auto physicalDevices = instance_.enumeratePhysicalDevices();
  if (physicalDevices.empty()) {
    throw std::runtime_error("No Vulkan-compatible physical devices found.");
  }

  // Try to find an integrated GPU
  const auto integrated_gpu =
      std::ranges::find_if(physicalDevices, [type](const auto &device) {
        return device.getProperties().deviceType == type;
      });

  if (integrated_gpu == physicalDevices.end()) {
    throw std::runtime_error("No integrated GPU found");
  }

  physical_device_ = *integrated_gpu;

  spdlog::info("Using integrated GPU: {}",
               physical_device_.getProperties().deviceName.data());
}

// ----------------------------------------------------------------------------
// Device
// ----------------------------------------------------------------------------

[[nodiscard]] vk::PhysicalDeviceVulkan12Features
check_vulkan_12_features(const vk::PhysicalDevice &physical_device) {
  // we want to query and check if uniformAndStorageBuffer8BitAccess is
  // supported before we can create this feature struct

  // vk::PhysicalDeviceFeatures2 features2;
  // physical_device.getFeatures2(&features2);

  // I need to enable these features for the 8-bit integer shader to work
  vk::PhysicalDeviceVulkan12Features vulkan12_features{
      // Allows the use of 8-bit integer types (int8_t, uint8_t) in storage
      // buffers. This feature enables shaders to read and write 8-bit integers
      // in storage buffers directly.
      .storageBuffer8BitAccess = true,

      // Extends the above capability to also allow 8-bit integer types in
      // uniform buffers (in addition to storage buffers). Uniform buffers
      // typically hold data shared between the CPU and GPU, such as constants
      // or frequently updated data.

      .uniformAndStorageBuffer8BitAccess = true,

      // Enables the use of 8-bit integer arithmetic (int8_t, uint8_t) in
      // shaders. This includes performing operations like addition,
      // subtraction, multiplication, and bitwise operations directly on 8-bit
      // integers.
      .shaderInt8 = true,

      // I got an error from VMA if I don't enable this:
      .bufferDeviceAddress = true,
  };

  vk::PhysicalDeviceFeatures2 features2{
      .pNext = &vulkan12_features,
  };

  physical_device.getFeatures2(&features2);

  return vulkan12_features;
}

void VulkanEngine::create_device(vk::QueueFlags queue_flags) {
  SPD_TRACE_FUNC

  if (!physical_device_) {
    throw std::runtime_error("Physical device is not valid");
  }

  const auto queueFamilyProperties =
      physical_device_.getQueueFamilyProperties();

  compute_queue_family_index_ = std::distance(
      queueFamilyProperties.begin(),
      std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
                   [queue_flags](const auto &qfp) {
                     return qfp.queueFlags & queue_flags;
                   }));

  if (compute_queue_family_index_ == queueFamilyProperties.size()) {
    throw std::runtime_error("No queue family supports compute operations.");
  }

  // Create a logical device with a compute queue
  constexpr float queuePriority = 1.0f;
  const vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
      .queueFamilyIndex = compute_queue_family_index_,
      .queueCount = 1,
      .pQueuePriorities = &queuePriority,
  };

  const auto vulkan_12_features = check_vulkan_12_features(physical_device_);

  const vk::DeviceCreateInfo deviceCreateInfo{
      .pNext = &vulkan_12_features,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &deviceQueueCreateInfo,
  };

  device_ = physical_device_.createDevice(deviceCreateInfo);

  // looks like I don't need this
  // VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);

  compute_queue_ = device_.getQueue(compute_queue_family_index_, 0);
}

// ----------------------------------------------------------------------------
// VMA allocator
// ----------------------------------------------------------------------------

void VulkanEngine::initialize_vma_allocator() {
  SPD_TRACE_FUNC

  if (!physical_device_ || !device_ || !instance_) {
    throw std::runtime_error(
        "Physical device, device, or instance is not valid");
  }

  // clang-format off
  const VmaVulkanFunctions vulkan_functions{
      .vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
      .vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr,
      .vkGetPhysicalDeviceProperties = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties,
      .vkGetPhysicalDeviceMemoryProperties = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties,
      .vkAllocateMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory,
      .vkFreeMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory,
      .vkMapMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkMapMemory,
      .vkUnmapMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkUnmapMemory,
      .vkFlushMappedMemoryRanges = VULKAN_HPP_DEFAULT_DISPATCHER.vkFlushMappedMemoryRanges,
      .vkInvalidateMappedMemoryRanges = VULKAN_HPP_DEFAULT_DISPATCHER.vkInvalidateMappedMemoryRanges,
      .vkBindBufferMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory,
      .vkBindImageMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory,
      .vkGetBufferMemoryRequirements = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements,
      .vkGetImageMemoryRequirements = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements,
      .vkCreateBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateBuffer,
      .vkDestroyBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer,
      .vkCreateImage = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage,
      .vkDestroyImage = VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage,
      .vkCmdCopyBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyBuffer,
      .vkGetBufferMemoryRequirements2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements2,
      .vkGetImageMemoryRequirements2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements2,
      .vkBindBufferMemory2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory2,
      .vkBindImageMemory2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory2,
      .vkGetPhysicalDeviceMemoryProperties2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties2,
      .vkGetDeviceBufferMemoryRequirements = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceBufferMemoryRequirements,
      .vkGetDeviceImageMemoryRequirements = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceImageMemoryRequirements,
  };
  // clang-format on

  const VmaAllocatorCreateInfo vma_allocator_create_info{
      .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice = physical_device_,
      .device = device_,
      .preferredLargeHeapBlockSize = 0, // Let VMA use default size
      .pAllocationCallbacks = nullptr,
      .pDeviceMemoryCallbacks = nullptr,
      .pHeapSizeLimit = nullptr,
      .pVulkanFunctions = &vulkan_functions,
      .instance = instance_,
      .vulkanApiVersion = VK_API_VERSION_1_3,
      .pTypeExternalMemoryHandleTypes = nullptr};

  if (vmaCreateAllocator(&vma_allocator_create_info, &g_vma_allocator) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create VMA allocator");
  }
}
// ----------------------------------------------------------------------------
// Helper functions
// 1) Subgroup size
// ----------------------------------------------------------------------------

uint32_t VulkanEngine::get_subgroup_size() const {
  vk::StructureChain<vk::PhysicalDeviceProperties2,
                     vk::PhysicalDeviceSubgroupProperties>
      propertyChain;

  physical_device_.getProperties2(
      &propertyChain.get<vk::PhysicalDeviceProperties2>());

  const auto subgroupProperties =
      propertyChain.get<vk::PhysicalDeviceSubgroupProperties>();

  return subgroupProperties.subgroupSize;
}
