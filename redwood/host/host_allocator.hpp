#pragma once

#include "../utils.hpp"

namespace cpu {

template <typename T>
struct HostAllocator {
  using value_type = T;

  T* allocate(std::size_t n) {
    if (n == 0) return nullptr;
    SPD_TRACE_FUNC
    spdlog::debug("HostAllocator: Allocating {} bytes", n * sizeof(T));
    return static_cast<T*>(operator new[](n * sizeof(T)));
  }

  void deallocate(T* p, std::size_t) noexcept {
    SPD_TRACE_FUNC
    operator delete[](p);
  }
};

}  // namespace cpu
