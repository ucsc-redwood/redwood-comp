#pragma once

#include <spdlog/spdlog.h>

#include "buffer.hpp"
#include "engine.hpp"

namespace vulkan {

// --------------------------------------------------------------------------
// VulkanMemoryResource for AppData
// Honestly, this should be called VulkanBackend
// --------------------------------------------------------------------------

class VulkanMemoryResource : public std::pmr::memory_resource {
 public:
  explicit VulkanMemoryResource(Engine& engine) : engine_(engine) {}

 protected:
  void* do_allocate(std::size_t bytes, std::size_t) override {
    auto buffer = engine_.buffer(bytes);
    return buffer->as<void*>();
  }

  void do_deallocate(void*, std::size_t, std::size_t) override {
    // NO OP
  }

  bool do_is_equal(const memory_resource& other) const noexcept override {
    // Equality means same engine instance
    auto* o = dynamic_cast<const VulkanMemoryResource*>(&other);
    return o && (&o->engine_ == &this->engine_);
  }

 private:
  Engine& engine_;
};

}  // namespace vulkan
