#pragma once

#include "../app_data.hpp"
#include "host_kernels.hpp"
#include "redwood/host/thread_pool.hpp"

namespace cpu {

namespace kernels {

namespace sparse {

void run_stage1(AppData& app_data, core::thread_pool& pool, size_t n_threads);
void run_stage2(AppData& app_data, core::thread_pool& pool, size_t n_threads);
void run_stage3(AppData& app_data, core::thread_pool& pool, size_t n_threads);
void run_stage4(AppData& app_data, core::thread_pool& pool, size_t n_threads);
void run_stage5(AppData& app_data, core::thread_pool& pool, size_t n_threads);
void run_stage6(AppData& app_data, core::thread_pool& pool, size_t n_threads);
void run_stage7(AppData& app_data, core::thread_pool& pool, size_t n_threads);
void run_stage8(AppData& app_data, core::thread_pool& pool, size_t n_threads);
void run_stage9(AppData& app_data, core::thread_pool& pool, size_t n_threads);

}

}  // namespace kernels

}  // namespace cpu