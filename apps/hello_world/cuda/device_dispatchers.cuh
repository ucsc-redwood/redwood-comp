#pragma once

#include "../app_data.hpp"

struct CUstream_st;
using cudaStream_t = CUstream_st *;

namespace cuda {

void run_stage1(AppData &app_data, cudaStream_t stream);

}  // namespace cuda
