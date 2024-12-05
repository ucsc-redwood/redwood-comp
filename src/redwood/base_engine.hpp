#pragma once

// A unified interface for all backends (CUDA, Vulkan, CPU)

class BaseEngine {
public:
  explicit BaseEngine() = default;

protected:
  virtual void destroy() = 0;
};
