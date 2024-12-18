#include <spdlog/spdlog.h>

#include "../app.hpp"
#include "app_data.hpp"
#include "host/host_dispatchers.hpp"
#include "redwood/backends.hpp"

constexpr auto n_iterations = 1000;
constexpr auto n_gpu_iterations = 1000;

// forward declarations
void run_vulkan_demo(const size_t input_size);
void run_cuda_demo(const size_t input_size);

void run_cpu_unpinned(const size_t input_size) {
  AppData app_data(std::pmr::new_delete_resource(), input_size);

  const auto n_threads = std::thread::hardware_concurrency();
  core::thread_pool pool(n_threads);

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < n_iterations; ++i) {
    cpu::run_stage1(app_data, pool, n_threads);
    cpu::run_stage2(app_data, pool, n_threads);
    cpu::run_stage3(app_data, pool, n_threads);
    cpu::run_stage4(app_data, pool, n_threads);
    cpu::run_stage5(app_data, pool, n_threads);
    cpu::run_stage6(app_data, pool, n_threads);
    cpu::run_stage7(app_data, pool, n_threads);

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

#ifdef REDWOOD_CUDA_BACKEND

#include <cuda_runtime.h>

#include "cuda/cu_dispatcher.cuh"
#include "redwood/cuda/cu_mem_resource.cuh"
#include "redwood/cuda/helpers.cuh"

void run_cuda_demo(const size_t input_size) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cuda::CudaMemoryResource mr;
  AppData app_data(&mr, input_size);

  // Allocate device memory for the sorting stage
  cuda::ImStorage im_storage(input_size);

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < n_gpu_iterations; ++i) {
    cuda::run_stage1(app_data, stream);
    cuda::run_stage2(app_data, im_storage, stream);
    cuda::run_stage3(app_data, im_storage, stream);
    cuda::run_stage4(app_data, stream);
    cuda::run_stage5(app_data, stream);
    cuda::run_stage6(app_data, stream);
    cuda::run_stage7(app_data, stream);

    if (i % 100 == 0) {
      std::cout << "." << std::flush;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "\nTotal time: " << duration.count() << "ms" << std::endl;
  std::cout << "Average time per iteration: "
            << static_cast<float>(duration.count()) / n_gpu_iterations << "ms"
            << std::endl;

  CUDA_CHECK(cudaStreamDestroy(stream));
}

#endif

#ifdef REDWOOD_VULKAN_BACKEND

#include "vulkan/vk_dispatcher.hpp"

void run_vulkan_demo(const size_t input_size) {
  vulkan::Engine engine;
  AppData app_data(engine.get_mr(), input_size);

  vulkan::TmpStorage tmp_storage(engine.get_mr(), input_size);
  vulkan::Dispatcher dispatcher(engine, app_data, tmp_storage);

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

    if (i % 100 == 0) {
      std::cout << "." << std::flush;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "\nTotal time: " << duration.count() << "ms" << std::endl;
  std::cout << "Average time per iteration: "
            << static_cast<float>(duration.count()) / n_gpu_iterations << "ms"
            << std::endl;

  spdlog::info("Vulkan Done.");
}

#endif

int main(int argc, char** argv) {
  INIT_APP("tree");

  run_cpu_unpinned(640 * 480);

  if constexpr (is_backend_enabled(BackendType::kCUDA)) {
    run_cuda_demo(640 * 480);
  }

  // if constexpr (is_backend_enabled(BackendType::kVulkan)) {
  //   run_vulkan_demo(640 * 480);
  // }

  spdlog::info("Done.");
  return EXIT_SUCCESS;
}
