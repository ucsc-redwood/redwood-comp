#pragma once

#include <cstddef>

class BaseBuffer {
 public:
  BaseBuffer() = delete;
  explicit BaseBuffer(size_t size) : size_(size) {}
  virtual ~BaseBuffer() = default;

 protected:
  virtual void allocate() = 0;
  virtual void free() = 0;

  const size_t size_;
  std::byte *mapped_data_ = nullptr;
};
