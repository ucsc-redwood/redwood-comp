#pragma once

#include "../app_data.hpp"
#include "redwood/host/host_dispatcher.hpp"
#include "redwood/host/thread_pool.hpp"

namespace cpu {

namespace kernels {

namespace sparse {

class Dispatcher final : public BaseDispatcher {
 public:
  Dispatcher(size_t n_pools, size_t n_threads_per_pool);

  void run_stage1(AppData& app_data,
                  core::thread_pool& pool,
                  size_t n_threads,
                  bool sync = false);
  void run_stage2(AppData& app_data,
                  core::thread_pool& pool,
                  size_t n_threads,
                  bool sync = false);
  void run_stage3(AppData& app_data,
                  core::thread_pool& pool,
                  size_t n_threads,
                  bool sync = false);
  void run_stage4(AppData& app_data,
                  core::thread_pool& pool,
                  size_t n_threads,
                  bool sync = false);
  void run_stage5(AppData& app_data,
                  core::thread_pool& pool,
                  size_t n_threads,
                  bool sync = false);
  void run_stage6(AppData& app_data,
                  core::thread_pool& pool,
                  size_t n_threads,
                  bool sync = false);
  void run_stage7(AppData& app_data,
                  core::thread_pool& pool,
                  size_t n_threads,
                  bool sync = false);
  void run_stage8(AppData& app_data,
                  core::thread_pool& pool,
                  size_t n_threads,
                  bool sync = false);
  void run_stage9(AppData& app_data,
                  core::thread_pool& pool,
                  size_t n_threads,
                  bool sync = false);

 private:
  std::vector<std::shared_ptr<AppData>> app_data_vec_;
};

// clang-format off
void run_stage1(AppData& app_data, core::thread_pool& pool, size_t n_threads, bool sync = false);
void run_stage2(AppData& app_data, core::thread_pool& pool, size_t n_threads, bool sync = false);
void run_stage3(AppData& app_data, core::thread_pool& pool, size_t n_threads, bool sync = false);
void run_stage4(AppData& app_data, core::thread_pool& pool, size_t n_threads, bool sync = false);
void run_stage5(AppData& app_data, core::thread_pool& pool, size_t n_threads, bool sync = false);
void run_stage6(AppData& app_data, core::thread_pool& pool, size_t n_threads, bool sync = false);
void run_stage7(AppData& app_data, core::thread_pool& pool, size_t n_threads, bool sync = false);
void run_stage8(AppData& app_data, core::thread_pool& pool, size_t n_threads, bool sync = false);
void run_stage9(AppData& app_data, core::thread_pool& pool, size_t n_threads, bool sync = false);
// clang-format on

}  // namespace sparse

}  // namespace kernels

}  // namespace cpu