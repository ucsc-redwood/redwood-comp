
#include "../../app.hpp"
#include "../app_data.hpp"
#include "../arg_max.hpp"
#include "host/host_dispatcher.hpp"
#include "redwood/backends.hpp"

// forward declare
void run_vulkan_demo();
void run_cuda_demo();

constexpr auto n_iterations = 1000;
constexpr auto n_gpu_iterations = 1000;

void run_cpu_unpinned() {
  auto mr = std::pmr::new_delete_resource();
  AppData app_data(mr);

  const auto n_threads = std::thread::hardware_concurrency();
  core::thread_pool pool(n_threads);

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < n_iterations; ++i) {
    cpu::run_stage1(app_data, pool, n_threads).wait();
    cpu::run_stage2(app_data, pool, n_threads).wait();
    cpu::run_stage3(app_data, pool, n_threads).wait();
    cpu::run_stage4(app_data, pool, n_threads).wait();
    cpu::run_stage5(app_data, pool, n_threads).wait();
    cpu::run_stage6(app_data, pool, n_threads).wait();
    cpu::run_stage7(app_data, pool, n_threads).wait();
    cpu::run_stage8(app_data, pool, n_threads).wait();
    cpu::run_stage9(app_data, pool, n_threads).wait();

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
}

#ifdef REDWOOD_VULKAN_BACKEND

#include "vulkan/vk_dispatcher.hpp"

void run_vulkan_demo() {
  vulkan::Engine engine;
  AppData app_data(engine.get_mr());

  vulkan::Dispatcher dispatcher(engine, app_data);

  auto seq = engine.sequence();

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < n_gpu_iterations; ++i) {
    dispatcher.run_stage1(seq.get());
    dispatcher.run_stage2(seq.get());
    dispatcher.run_stage3(seq.get());
    dispatcher.run_stage4(seq.get());
    dispatcher.run_stage5(seq.get());
    dispatcher.run_stage6(seq.get());
    dispatcher.run_stage7(seq.get());
    dispatcher.run_stage8(seq.get());
    dispatcher.run_stage9(seq.get());

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
}

#endif

#ifdef REDWOOD_CUDA_BACKEND

#include <cuda_runtime.h>

#include "cuda/cu_dispatcher.cuh"
#include "redwood/cuda/cu_mem_resource.cuh"

void run_cuda_demo() {
  cuda::CudaMemoryResource mr;
  AppData app_data(&mr);

  constexpr auto n_concurrent = 1;
  cuda::Dispatcher dispatcher(app_data, n_concurrent);

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < n_gpu_iterations; ++i) {
    dispatcher.run_stage1(0);
    dispatcher.run_stage2(0);
    dispatcher.run_stage3(0);
    dispatcher.run_stage4(0);
    dispatcher.run_stage5(0);
    dispatcher.run_stage6(0);
    dispatcher.run_stage7(0);
    dispatcher.run_stage8(0);
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
}

#endif

int main(int argc, char** argv) {
  INIT_APP("cifar_dense");

  run_cpu_unpinned();

  if constexpr (is_backend_enabled(BackendType::kVulkan)) {
    run_vulkan_demo();
  }

  if constexpr (is_backend_enabled(BackendType::kCUDA)) {
    run_cuda_demo();
  }

  spdlog::info("Done");
  return EXIT_SUCCESS;
}
