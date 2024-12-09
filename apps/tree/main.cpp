#include <spdlog/spdlog.h>

#include "../cli_to_config.hpp"
#include "../read_config.hpp"
#include "app_data.hpp"
#include "host/host_dispatchers.hpp"
#include "redwood/backends.hpp"

// forward declarations
void run_vulkan_demo(const size_t input_size);
void run_cuda_demo(const size_t input_size);

void print_stats(const AppData& app_data) {
  spdlog::info("num_unique = {}", app_data.get_n_unique());
  spdlog::info("num_brt = {}", app_data.get_n_brt_nodes());
  spdlog::info("num_octree_nodes = {}", app_data.get_n_octree_nodes());
}

void run_cpu_demo(const std::vector<int>& cores, const size_t input_size) {
  AppData app_data(std::pmr::new_delete_resource(), input_size);

  auto n_threads = cores.size();

  core::thread_pool pool(cores);
  cpu::run_stage1(app_data, pool, n_threads);
  cpu::run_stage2(app_data, pool, n_threads);

  if (!std::ranges::is_sorted(app_data.u_morton_keys)) {
    spdlog::error("Morton keys are not sorted");
  }

  cpu::run_stage3(app_data, pool, n_threads);
  cpu::run_stage4(app_data, pool, n_threads);
  cpu::run_stage5(app_data, pool, n_threads);
  cpu::run_stage6(app_data, pool, n_threads);
  cpu::run_stage7(app_data, pool, n_threads);

  print_stats(app_data);
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

  cuda::run_stage1(app_data, stream);
  cuda::run_stage2(app_data, im_storage, stream);

  if (std::ranges::is_sorted(app_data.u_morton_keys)) {
    spdlog::info("Morton keys are sorted");
  } else {
    spdlog::error("Morton keys are not sorted");
  }

  cuda::run_stage3(app_data, im_storage, stream);
  cuda::run_stage4(app_data, stream);
  cuda::run_stage5(app_data, stream);

  // peek 10 edge count
  for (auto i = 0; i < 10; ++i) {
    spdlog::info("edge_count[{}] = {}", i, app_data.u_edge_count[i]);
  }

  cuda::run_stage6(app_data, stream);

  // peek 10 edge offset
  for (auto i = 0; i < 10; ++i) {
    spdlog::info("edge_offset[{}] = {}", i, app_data.u_edge_offset[i]);
  }

  cuda::run_stage7(app_data, stream);

  print_stats(app_data);

  CUDA_CHECK(cudaStreamDestroy(stream));
}

#endif

#ifdef REDWOOD_VULKAN_BACKEND

#include "vulkan/vk_dispatcher.hpp"

void run_vulkan_demo(const size_t input_size) {
  vulkan::Engine engine;
  AppData app_data(engine.get_mr(), input_size);

  vulkan::Dispatcher dispatcher(engine, app_data);

  auto seq = engine.sequence();

  // dispatcher.run_stage1(seq.get());
  // dispatcher.run_stage2(seq.get());
  // dispatcher.run_stage3(seq.get());
  // dispatcher.run_stage4(seq.get());
  // dispatcher.run_stage5(seq.get());
  // dispatcher.run_stage6(seq.get());
  // dispatcher.run_stage7(seq.get());

  // print_stats(app_data);

  spdlog::info("Vulkan Done.");
}

#endif

int main(int argc, char** argv) {
  auto config = helpers::init_demo(argc, argv);
  auto small_cores = helpers::get_cores_by_type(config["cpu_info"], "small");
  auto medium_cores = helpers::get_cores_by_type(config["cpu_info"], "medium");
  auto big_cores = helpers::get_cores_by_type(config["cpu_info"], "big");

  assert(!small_cores.empty());

  spdlog::set_level(spdlog::level::trace);

  run_cpu_demo(small_cores, 640 * 480);

  if constexpr (is_backend_enabled(BackendType::kCUDA)) {
    run_cuda_demo(640 * 480);
  }

  if constexpr (is_backend_enabled(BackendType::kVulkan)) {
    run_vulkan_demo(640 * 480);
  }

  spdlog::info("Done.");

  return EXIT_SUCCESS;
}
