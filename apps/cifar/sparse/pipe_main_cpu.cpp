#include "../../app.hpp"
#include "app_data.hpp"
#include "host/host_dispatcher.hpp"

int main(int argc, char **argv) {
  INIT_APP("pipe-cifar-sparse");

  auto mr = std::pmr::new_delete_resource();
  AppData app_data(mr);

  // core::thread_pool small_pool(g_small_cores);
  // core::thread_pool medium_pool(g_medium_cores);
  // core::thread_pool large_pool(g_big_cores);

  constexpr auto n_iterations = 10000;

  const auto n_threads = std::thread::hardware_concurrency();
  core::thread_pool unpinned_pool(n_threads);

  auto start = std::chrono::high_resolution_clock::now();

  std::cout << "Running " << n_iterations << " iterations" << std::endl;
  for (size_t i = 0; i < n_iterations; ++i) {
    cpu::kernels::sparse::run_stage1(app_data, unpinned_pool, n_threads, true);
    cpu::kernels::sparse::run_stage2(app_data, unpinned_pool, n_threads, true);
    cpu::kernels::sparse::run_stage3(app_data, unpinned_pool, n_threads, true);
    cpu::kernels::sparse::run_stage4(app_data, unpinned_pool, n_threads, true);
    cpu::kernels::sparse::run_stage5(app_data, unpinned_pool, n_threads, true);
    cpu::kernels::sparse::run_stage6(app_data, unpinned_pool, n_threads, true);
    cpu::kernels::sparse::run_stage7(app_data, unpinned_pool, n_threads, true);
    cpu::kernels::sparse::run_stage8(app_data, unpinned_pool, n_threads, true);
    cpu::kernels::sparse::run_stage9(app_data, unpinned_pool, n_threads, true);

    if (i % 100 == 0) {
      std::cout << "." << std::flush;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "\nTotal time: " << duration.count() << "ms" << std::endl;
  std::cout << "Average time per iteration: "
            << static_cast<float>(duration.count()) / n_iterations << "ms"
            << std::endl;
  std::cout << "Throughput: "
            << static_cast<float>(n_iterations) * 1000 / duration.count()
            << " iterations/s" << std::endl;

  return EXIT_SUCCESS;
}
