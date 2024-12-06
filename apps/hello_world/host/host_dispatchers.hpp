#pragma once

#include <spdlog/spdlog.h>

#include <future>

#include "../app_data.hpp"
#include "redwood/host/thread_pool.hpp"

namespace cpu {

namespace v1 {

[[nodiscard]] std::future<void> run_stage1(AppData &app_data);

}  // namespace v1

namespace v2 {

[[nodiscard]] core::multi_future<void> run_stage1(AppData &app_data,
                                                  core::thread_pool &pool,
                                                  const size_t n_threads);

}  // namespace v2

}  // namespace cpu
