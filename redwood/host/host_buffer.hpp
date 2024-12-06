#pragma once

#include "../utils.hpp"
#include "redwood/base_buffer.hpp"

namespace cpu {

class HostBuffer : public BaseBuffer {
 public:
  explicit HostBuffer(const size_t size) : BaseBuffer(size) { allocate(); }

  ~HostBuffer() override { free(); }

 protected:
  void allocate() override {
    SPD_TRACE_FUNC;
    mapped_data_ = new std::byte[size_];
  }
  void free() override {
    SPD_TRACE_FUNC;
    delete[] mapped_data_;
  }
};

}  // namespace cpu
