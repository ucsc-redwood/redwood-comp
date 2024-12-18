#include "../../app.hpp"
#include "app_data.hpp"
#include "host/host_dispatcher.hpp"
#include "redwood/backends.hpp"

constexpr auto n_iterations = 10000;
constexpr auto n_gpu_iterations = 1000;

void run_cuda_only();
void run_vulkan_only();

void run_cpu_unpinned() {
  auto mr = std::pmr::new_delete_resource();
  AppData app_data(mr);

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
}

#ifdef REDWOOD_CUDA_BACKEND

#include "cuda/cu_dispatcher.cuh"
#include "redwood/cuda/cu_mem_resource.cuh"

void run_cuda_only() {
  cuda::CudaMemoryResource mr;
  AppData app_data(&mr);

  cuda::Dispatcher dispatcher(app_data, 1);

  auto start = std::chrono::high_resolution_clock::now();

  std::cout << "Running " << n_gpu_iterations << " iterations" << std::endl;
  for (size_t i = 0; i < n_gpu_iterations; ++i) {
    dispatcher.run_stage1(0, true);
    dispatcher.run_stage2(0, true);
    dispatcher.run_stage3(0, true);
    dispatcher.run_stage4(0, true);
    dispatcher.run_stage5(0, true);
    dispatcher.run_stage6(0, true);
    dispatcher.run_stage7(0, true);
    dispatcher.run_stage8(0, true);
    dispatcher.run_stage9(0, true);

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
}

#endif

#ifdef REDWOOD_VULKAN_BACKEND

#include "vulkan/vk_dispatchers.hpp"

void run_vulkan_only() {
  vulkan::Engine engine;
  AppData app_data(engine.get_mr());

  vulkan::Dispatcher dispatcher(engine, app_data);

  auto seq = engine.sequence();

  auto start = std::chrono::high_resolution_clock::now();

  std::cout << "Running " << n_gpu_iterations << " iterations" << std::endl;
  for (size_t i = 0; i < n_gpu_iterations; ++i) {
    dispatcher.run_stage1(seq.get(), true);
    dispatcher.run_stage2(seq.get(), true);
    dispatcher.run_stage3(seq.get(), true);
    dispatcher.run_stage4(seq.get(), true);
    dispatcher.run_stage5(seq.get(), true);
    dispatcher.run_stage6(seq.get(), true);
    dispatcher.run_stage7(seq.get(), true);
    dispatcher.run_stage8(seq.get(), true);
    dispatcher.run_stage9(seq.get(), true);

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
}

#endif

int main(int argc, char **argv) {
  INIT_APP("pipe-cifar-sparse");

  spdlog::info("Running CPU unpinned");
  run_cpu_unpinned();

  if constexpr (is_backend_enabled(BackendType::kCUDA)) {
    spdlog::info("Running CUDA only");
    run_cuda_only();
  }

  if constexpr (is_backend_enabled(BackendType::kVulkan)) {
    spdlog::info("Running Vulkan only");
    run_vulkan_only();
  }

  return EXIT_SUCCESS;
}