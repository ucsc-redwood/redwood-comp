#pragma once

#include "buffer.hpp"
#include "engine.hpp"

namespace vulkan {

// class VulkanMemoryResource : public std::pmr::memory_resource {
//  public:
//   explicit VulkanMemoryResource(Engine& eng) : engine_(eng) {}

//  protected:
//   void* do_allocate(std::size_t bytes, std::size_t) override {
//     if (allocated_) {
//       throw std::bad_alloc();
//     }

//     buffer_ = engine_.buffer(bytes);

//     allocated_ = true;
//     return buffer_->as<void*>();
//   }

//   void do_deallocate(void*, std::size_t, std::size_t) override {
//     // NO OP
//   }

//   bool do_is_equal(const memory_resource& other) const noexcept override {
//     // Equality means same engine instance
//     auto* o = dynamic_cast<const VulkanMemoryResource*>(&other);
//     return o && (&o->engine_ == &this->engine_);
//   }

//  private:
//   Engine& engine_;
//   std::shared_ptr<Buffer> buffer_;

//   bool allocated_ = false;
// };

class VulkanMemoryResource : public std::pmr::memory_resource {
 public:
  explicit VulkanMemoryResource() : engine_() {}

 protected:
  void* do_allocate(std::size_t bytes, std::size_t) override {
    buffer_ = engine_.buffer(bytes);
    return buffer_->as<void*>();
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
  Engine engine_;
  std::shared_ptr<Buffer> buffer_;
};

}  // namespace vulkan
