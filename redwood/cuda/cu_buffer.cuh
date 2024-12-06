#pragma once

#include "../buffer.hpp"

namespace cuda {

class Buffer : public BaseBuffer {
public:
  Buffer() = delete;
  explicit Buffer(const size_t size) : BaseBuffer(size) { allocate(); }

  ~Buffer() override { free(); }

protected:
  void allocate() override;
  void free() override;
};

} // namespace cuda
