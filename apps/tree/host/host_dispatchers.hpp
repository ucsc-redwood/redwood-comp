#pragma once

#include "../app_data.hpp"
#include "redwood/host/thread_pool.hpp"

namespace cpu {

// input -> morton
void run_stage1(AppData& app_data,
                core::thread_pool& pool,
                const size_t n_threads);

// morton -> sorted morton
void run_stage2(AppData& app_data,
                core::thread_pool& pool,
                const size_t n_threads);

// sorted morton -> unique morton
void run_stage3(AppData& app_data,
                core::thread_pool& pool,
                const size_t n_threads);

// unique morton -> brt
void run_stage4(AppData& app_data,
                core::thread_pool& pool,
                const size_t n_threads);

// brt -> edge count
void run_stage5(AppData& app_data,
                core::thread_pool& pool,
                const size_t n_threads);

// edge count -> edge offset
void run_stage6(AppData& app_data,
                core::thread_pool& pool,
                const size_t n_threads);

// *everything above* -> octree
void run_stage7(AppData& app_data,
                core::thread_pool& pool,
                const size_t n_threads);

}  // namespace cpu