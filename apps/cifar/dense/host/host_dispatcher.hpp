#pragma once

#include "../../app_data.hpp"
#include "redwood/host/thread_pool.hpp"

namespace cpu {

[[nodiscard]] core::multi_future<void> run_stage1(AppData &app_data,
                                                  core::thread_pool &pool,
                                                  size_t n_threads);

[[nodiscard]] core::multi_future<void> run_stage2(AppData &app_data,
                                                  core::thread_pool &pool,
                                                  size_t n_threads);

[[nodiscard]] core::multi_future<void> run_stage3(AppData &app_data,
                                                  core::thread_pool &pool,
                                                  size_t n_threads);

[[nodiscard]] core::multi_future<void> run_stage4(AppData &app_data,
                                                  core::thread_pool &pool,
                                                  size_t n_threads);

[[nodiscard]] core::multi_future<void> run_stage5(AppData &app_data,
                                                  core::thread_pool &pool,
                                                  size_t n_threads);

[[nodiscard]] core::multi_future<void> run_stage6(AppData &app_data,
                                                  core::thread_pool &pool,
                                                  size_t n_threads);

[[nodiscard]] core::multi_future<void> run_stage7(AppData &app_data,
                                                  core::thread_pool &pool,
                                                  size_t n_threads);

[[nodiscard]] core::multi_future<void> run_stage8(AppData &app_data,
                                                  core::thread_pool &pool,
                                                  size_t n_threads);

[[nodiscard]] core::multi_future<void> run_stage9(AppData &app_data,
                                                  core::thread_pool &pool,
                                                  size_t n_threads);

inline void run_stage1_sync(AppData &app_data,
                            core::thread_pool &pool,
                            size_t n_threads) {
  run_stage1(app_data, pool, n_threads).wait();
}
inline void run_stage2_sync(AppData &app_data,
                            core::thread_pool &pool,
                            size_t n_threads) {
  run_stage2(app_data, pool, n_threads).wait();
}
inline void run_stage3_sync(AppData &app_data,
                            core::thread_pool &pool,
                            size_t n_threads) {
  run_stage3(app_data, pool, n_threads).wait();
}
inline void run_stage4_sync(AppData &app_data,
                            core::thread_pool &pool,
                            size_t n_threads) {
  run_stage4(app_data, pool, n_threads).wait();
}
inline void run_stage5_sync(AppData &app_data,
                            core::thread_pool &pool,
                            size_t n_threads) {
  run_stage5(app_data, pool, n_threads).wait();
}
inline void run_stage6_sync(AppData &app_data,
                            core::thread_pool &pool,
                            size_t n_threads) {
  run_stage6(app_data, pool, n_threads).wait();
}
inline void run_stage7_sync(AppData &app_data,
                            core::thread_pool &pool,
                            size_t n_threads) {
  run_stage7(app_data, pool, n_threads).wait();
}
inline void run_stage8_sync(AppData &app_data,
                            core::thread_pool &pool,
                            size_t n_threads) {
  run_stage8(app_data, pool, n_threads).wait();
}
inline void run_stage9_sync(AppData &app_data,
                            core::thread_pool &pool,
                            size_t n_threads) {
  run_stage9(app_data, pool, n_threads).wait();
}

}  // namespace cpu
