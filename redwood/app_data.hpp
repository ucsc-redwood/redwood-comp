#pragma once

#include "cuda/cu_buffer_typed.cuh"

#include <memory>

namespace app {

// --------------------------------------------------------------------------
// First Application: Demo
// --------------------------------------------------------------------------

namespace demo {

// --------------------------------------------------------------------------
// Data needed for the demo
// --------------------------------------------------------------------------

class AppData {
public:
  explicit AppData() {
    u_buffer_A = std::make_shared<cuda::TypedBuffer<int>>(1024);
    u_buffer_B = std::make_shared<cuda::TypedBuffer<int>>(1024);
    u_buffer_C = std::make_shared<cuda::TypedBuffer<int>>(1024);
  }

  ~AppData() = default;

private:
  std::shared_ptr<cuda::TypedBuffer<int>> u_buffer_A;
  std::shared_ptr<cuda::TypedBuffer<int>> u_buffer_B;
  std::shared_ptr<cuda::TypedBuffer<int>> u_buffer_C;
};

} // namespace demo

// --------------------------------------------------------------------------
// Kernels
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Dispatcher
// the purpose of dispatcher is to launch
// --------------------------------------------------------------------------

} // namespace app
