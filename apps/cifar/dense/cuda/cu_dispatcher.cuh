#pragma once

#include "../../app_data.hpp"

namespace cuda {


// clang-format off
void run_stage1(AppData &app_data, const cudaStream_t stream, bool sync = false);
void run_stage2(AppData &app_data, const cudaStream_t stream, bool sync = false);
void run_stage3(AppData &app_data, const cudaStream_t stream, bool sync = false);
void run_stage4(AppData &app_data, const cudaStream_t stream, bool sync = false);
void run_stage5(AppData &app_data, const cudaStream_t stream, bool sync = false);
void run_stage6(AppData &app_data, const cudaStream_t stream, bool sync = false);
void run_stage7(AppData &app_data, const cudaStream_t stream, bool sync = false);
void run_stage8(AppData &app_data, const cudaStream_t stream, bool sync = false);
void run_stage9(AppData &app_data, const cudaStream_t stream, bool sync = false);
// clang-format on

}  // namespace cuda
