#pragma once

#include "../app_data.hpp"
#include "redwood/cuda/cu_dispatcher.cuh"

namespace cuda {

void run_stage1(CuDispatcher &dispatcher, AppData &app_data);

}  // namespace cuda
