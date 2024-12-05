#pragma once

#include <cstdint>

// USM (Unified Shared Memory) buffer, providing a unified interface for all
// backends
class UsmBuffer {
public:

  explicit UsmBuffer(uint32_t size);

  virtual ~UsmBuffer() = default;

protected:
  uint32_t size;
};
