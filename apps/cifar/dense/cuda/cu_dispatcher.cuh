#pragma once

#include "../../app_data.hpp"

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

// // clang-format off
// void run_stage1(AppData &app_data, const cudaStream_t stream, bool sync =
// false); void run_stage2(AppData &app_data, const cudaStream_t stream, bool
// sync = false); void run_stage3(AppData &app_data, const cudaStream_t stream,
// bool sync = false); void run_stage4(AppData &app_data, const cudaStream_t
// stream, bool sync = false); void run_stage5(AppData &app_data, const
// cudaStream_t stream, bool sync = false); void run_stage6(AppData &app_data,
// const cudaStream_t stream, bool sync = false); void run_stage7(AppData
// &app_data, const cudaStream_t stream, bool sync = false); void
// run_stage8(AppData &app_data, const cudaStream_t stream, bool sync = false);
// void run_stage9(AppData &app_data, const cudaStream_t stream, bool sync =
// false);
// // clang-format on

}  // namespace cuda
