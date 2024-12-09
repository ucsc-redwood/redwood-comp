#include "../../cli_to_config.hpp"
#include "../../read_config.hpp"
#include "../app_data.hpp"
#include "host/host_dispatcher.hpp"
#include "redwood/backends.hpp"

// forward declare
void run_vulkan_demo();
void run_cuda_demo();

[[nodiscard]] int arg_max(const float* ptr) {
  const auto max_index = std::distance(
      ptr, std::ranges::max_element(ptr, ptr + model::kLinearOutFeatures));

  return max_index;
}

void print_prediction(const int max_index) {
  static const std::unordered_map<int, std::string_view> class_names{
      {0, "airplanes"},
      {1, "cars"},
      {2, "birds"},
      {3, "cats"},
      {4, "deer"},
      {5, "dogs"},
      {6, "frogs"},
      {7, "horses"},
      {8, "ships"},
      {9, "trucks"}};

  std::cout << "Predicted Image: ";
  std::cout << (class_names.contains(max_index) ? class_names.at(max_index)
                                                : "Unknown");
  std::cout << std::endl;
}

void run_cpu_demo(const std::vector<int>& cores, const size_t n_threads) {
  auto mr = std::pmr::new_delete_resource();
  AppData app_data(mr);

  core::thread_pool pool(cores);

  cpu::run_stage1(app_data, pool, n_threads).wait();
  cpu::run_stage2(app_data, pool, n_threads).wait();
  cpu::run_stage3(app_data, pool, n_threads).wait();
  cpu::run_stage4(app_data, pool, n_threads).wait();
  cpu::run_stage5(app_data, pool, n_threads).wait();
  cpu::run_stage6(app_data, pool, n_threads).wait();
  cpu::run_stage7(app_data, pool, n_threads).wait();
  cpu::run_stage8(app_data, pool, n_threads).wait();
  cpu::run_stage9(app_data, pool, n_threads).wait();

  print_prediction(arg_max(app_data.u_linear_out.data()));
}

#ifdef REDWOOD_VULKAN_BACKEND

#include "vulkan/vk_dispatcher.hpp"

void run_vulkan_demo() {
  vulkan::Engine engine;
  AppData app_data(engine.get_mr());

  vulkan::Dispatcher dispatcher(engine, app_data);

  auto seq = engine.sequence();

  dispatcher.run_stage1(seq.get());
  dispatcher.run_stage2(seq.get());
  dispatcher.run_stage3(seq.get());
  dispatcher.run_stage4(seq.get());
  dispatcher.run_stage5(seq.get());
  dispatcher.run_stage6(seq.get());
  dispatcher.run_stage7(seq.get());
  dispatcher.run_stage8(seq.get());
  dispatcher.run_stage9(seq.get());

  print_prediction(arg_max(app_data.u_linear_out.data()));
}

#endif

#ifdef REDWOOD_CUDA_BACKEND

#include <cuda_runtime.h>

#include "cuda/cu_dispatcher.cuh"
#include "redwood/cuda/cu_mem_resource.cuh"
#include "redwood/cuda/helpers.cuh"

void run_cuda_demo() {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cuda::CudaMemoryResource mr;

  AppData app_data(&mr);
  cuda::run_stage1(app_data, stream);
  cuda::run_stage2(app_data, stream);
  cuda::run_stage3(app_data, stream);
  cuda::run_stage4(app_data, stream);
  cuda::run_stage5(app_data, stream);
  cuda::run_stage6(app_data, stream);
  cuda::run_stage7(app_data, stream);
  cuda::run_stage8(app_data, stream);
  cuda::run_stage9(app_data, stream);

  print_prediction(arg_max(app_data.u_linear_out.data()));

  CUDA_CHECK(cudaStreamDestroy(stream));
}

#endif

int main(int argc, char** argv) {
  auto config = helpers::init_demo(argc, argv);
  auto small_cores = helpers::get_cores_by_type(config["cpu_info"], "small");
  auto medium_cores = helpers::get_cores_by_type(config["cpu_info"], "medium");
  auto big_cores = helpers::get_cores_by_type(config["cpu_info"], "big");

  assert(!small_cores.empty());

  spdlog::set_level(spdlog::level::trace);

  run_cpu_demo(small_cores, small_cores.size());

  if constexpr (is_backend_enabled(BackendType::kVulkan)) {
    spdlog::info("Vulkan backend is enabled");
    run_vulkan_demo();
  }

  if constexpr (is_backend_enabled(BackendType::kCUDA)) {
    spdlog::info("CUDA backend is enabled");
    run_cuda_demo();
  }

  spdlog::info("Done");
  return EXIT_SUCCESS;
}
