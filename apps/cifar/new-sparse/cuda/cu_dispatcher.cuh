#pragma once

#include <cuda_runtime_api.h>

#include "../app_data.hpp"

namespace cuda {

struct Dispatcher {
  explicit Dispatcher(AppData &app_data, size_t n_concurrent);
  ~Dispatcher();

  void run_stage1(size_t stream_id, bool sync = false);
  void run_stage2(size_t stream_id, bool sync = false);
  void run_stage3(size_t stream_id, bool sync = false);
  void run_stage4(size_t stream_id, bool sync = false);
  void run_stage5(size_t stream_id, bool sync = false);
  void run_stage6(size_t stream_id, bool sync = false);
  void run_stage7(size_t stream_id, bool sync = false);
  void run_stage8(size_t stream_id, bool sync = false);
  void run_stage9(size_t stream_id, bool sync = false);

 private:
  AppData &app_data;
  std::vector<cudaStream_t> streams;
};

}  // namespace cuda
