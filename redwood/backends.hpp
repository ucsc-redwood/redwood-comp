#pragma once

enum class BackendType { kCPU = 1 << 0, kCUDA = 1 << 1, kVulkan = 1 << 2 };

constexpr BackendType operator|(BackendType a, BackendType b) {
  return static_cast<BackendType>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr BackendType operator&(BackendType a, BackendType b) {
  return static_cast<BackendType>(static_cast<int>(a) & static_cast<int>(b));
}

constexpr BackendType kBackendType = BackendType::kCPU
#ifdef REDWOOD_CUDA_BACKEND
                                     | BackendType::kCUDA
#endif
#ifdef REDWOOD_VULKAN_BACKEND
                                     | BackendType::kVulkan
#endif
    ;

// ----------------------------------------------------------------------------
// Check if a backend is enabled (compile-time)
// ----------------------------------------------------------------------------

[[nodiscard]] constexpr bool is_backend_enabled(const BackendType backend) {
  return (kBackendType & backend) == backend;
}
