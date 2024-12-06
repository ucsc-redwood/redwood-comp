#pragma once

#include "../app_data.hpp"

namespace cuda {

void run_stage1(AppData &app_data, cudaStream_t stream);

}  // namespace cuda
