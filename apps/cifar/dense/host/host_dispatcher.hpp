#pragma once

#include "../app_data.hpp"
#include "redwood/host/thread_pool.hpp"

namespace cpu {

[[nodiscard]] core::multi_future<void> run_stage1(AppData &app_data,
                                                  core::thread_pool &pool,
                                                  const size_t n_threads);

}  // namespace cpu
