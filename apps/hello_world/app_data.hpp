#pragma once

#include "redwood/cuda/cuda_engine.hpp"

namespace cuda {

template <typename T> using BufferPtr = std::shared_ptr<TypedBuffer<T>>;

struct AppData {
  explicit AppData(Engine &engine, const size_t n) {
    input_a = engine.typed_buffer<int>(n);
    input_b = engine.typed_buffer<int>(n);
    output = engine.typed_buffer<int>(n);
  }

  BufferPtr<int> input_a;
  BufferPtr<int> input_b;
  BufferPtr<int> output;
};

} // namespace cuda
