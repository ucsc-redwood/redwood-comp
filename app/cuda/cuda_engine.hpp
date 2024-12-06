#pragma once

#include "cu_buffer_typed.cuh"

#include <memory>
#include <vector>

namespace cuda {

class Engine {
public:
  Engine(const bool manage_resources = true)
      : manage_resources_(manage_resources) {}

  template <typename T, typename... Args>
    requires std::is_constructible_v<cuda::TypedBuffer<T>, Args...>
  [[nodiscard]] std::shared_ptr<cuda::TypedBuffer<T>>
  typed_buffer(Args &&...args) {
    const auto buffer = std::make_shared<cuda::TypedBuffer<T>>(args...);

    if (manage_resources_) {
      buffers_.push_back(buffer);
    }

    return buffer;
  }

  

private:
  std::vector<std::weak_ptr<cuda::Buffer>> buffers_;

  const bool manage_resources_ = true;
};

} // namespace cuda
