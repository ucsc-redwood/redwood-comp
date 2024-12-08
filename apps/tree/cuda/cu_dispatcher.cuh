#pragma once

#include "../app_data.hpp"
#include "im_storage.cuh"

namespace cuda {

void run_stage1(AppData &app_data, cudaStream_t stream);

// special
void run_stage2(AppData &app_data, ImStorage &im_storage, cudaStream_t stream);

// special
void run_stage3(AppData &app_data, ImStorage &im_storage, cudaStream_t stream);

void run_stage4(AppData &app_data, cudaStream_t stream);
void run_stage5(AppData &app_data, cudaStream_t stream);
void run_stage6(AppData &app_data, cudaStream_t stream);
void run_stage7(AppData &app_data, cudaStream_t stream);

}  // namespace cuda
