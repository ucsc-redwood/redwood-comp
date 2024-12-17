#include "../../app.hpp"
#include "app_data.hpp"
#include "host/host_dispatcher.hpp"
#include "redwood/backends.hpp"
#include "redwood/host/thread_pool.hpp"

[[nodiscard]] inline int arg_max(const float* ptr) {
  const auto max_index =
      std::distance(ptr, std::ranges::max_element(ptr, ptr + 10));

  return max_index;
}

inline void print_prediction(const int max_index) {
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

void run_cpu_demo_v1() {
  auto mr = std::pmr::new_delete_resource();
  AppData app_data(mr);

  core::thread_pool pool(g_small_cores);
  const auto n_threads = g_small_cores.size();

  cpu::kernels::sparse::run_stage1(app_data, pool, n_threads, true);
  cpu::kernels::sparse::run_stage2(app_data, pool, n_threads, true);
  cpu::kernels::sparse::run_stage3(app_data, pool, n_threads, true);
  cpu::kernels::sparse::run_stage4(app_data, pool, n_threads, true);
  cpu::kernels::sparse::run_stage5(app_data, pool, n_threads, true);
  cpu::kernels::sparse::run_stage6(app_data, pool, n_threads, true);
  cpu::kernels::sparse::run_stage7(app_data, pool, n_threads, true);
  cpu::kernels::sparse::run_stage8(app_data, pool, n_threads, true);
  cpu::kernels::sparse::run_stage9(app_data, pool, n_threads, true);

  print_prediction(arg_max(app_data.u_linear_output.data()));
}

#ifdef REDWOOD_CUDA_BACKEND

#include "cuda/cu_dispatcher.cuh"
#include "redwood/cuda/cu_mem_resource.cuh"

void run_cuda_demo_v1() {
  cuda::CudaMemoryResource mr;
  AppData app_data(&mr);

  cuda::Dispatcher dispatcher(app_data, g_small_cores.size());

  dispatcher.run_stage1(0, true);
  dispatcher.run_stage2(0, true);
  dispatcher.run_stage3(0, true);
  dispatcher.run_stage4(0, true);
  dispatcher.run_stage5(0, true);
  dispatcher.run_stage6(0, true);
  dispatcher.run_stage7(0, true);
  dispatcher.run_stage8(0, true);
  dispatcher.run_stage9(0, true);

  print_prediction(arg_max(app_data.u_linear_output.data()));
}

#endif

int main(int argc, char** argv) {
  INIT_APP("cifar-sparse");

  run_cpu_demo_v1();

  if constexpr (is_backend_enabled(BackendType::kCUDA)) {
    run_cuda_demo_v1();
  }

  return 0;
}
