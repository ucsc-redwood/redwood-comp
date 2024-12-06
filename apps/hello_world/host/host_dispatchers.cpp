#include "host_dispatchers.hpp"

#include "host_kernels.hpp"

namespace cpu {

namespace v1 {

[[nodiscard]] std::future<void> run_stage1(AppData &app_data) {
  spdlog::debug("CPU kernel 'vector_add', n = {}", app_data.n);

  return std::async(std::launch::async, [&]() {
    cpu::kernels::vector_add(app_data.u_input_a.data(),
                             app_data.u_input_b.data(),
                             app_data.u_output.data(),
                             0,
                             app_data.n);
  });
}

}  // namespace v1

namespace v2 {

[[nodiscard]] core::multi_future<void> run_stage1(AppData &app_data,
                                                  core::thread_pool &pool,
                                                  const size_t n_threads) {
  spdlog::debug(
      "CPU kernel 'vector_add', n = {}, threads = {}", app_data.n, n_threads);

  return pool.submit_blocks(
      static_cast<size_t>(0),
      app_data.n,
      [&](const size_t start, const size_t end) {
        cpu::kernels::vector_add(app_data.u_input_a.data(),
                                 app_data.u_input_b.data(),
                                 app_data.u_output.data(),
                                 start,
                                 end);
      },
      n_threads);
}

}  // namespace v2

}  // namespace cpu
