#pragma once

#include "../base_engine.hpp"

class CudaEngine : public BaseEngine {
public:
  explicit CudaEngine();

  ~CudaEngine() { destroy(); }

protected:
  void destroy() override;
};
