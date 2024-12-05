#pragma once

#include "../usm_buffer.hpp"

class CudaBuffer : public UsmBuffer {
public:
  explicit CudaBuffer(uint32_t size);

  ~CudaBuffer() override;

private:
  void *u_data;
};
