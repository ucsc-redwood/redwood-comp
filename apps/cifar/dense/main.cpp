#include "../../cli_to_config.hpp"
#include "../../read_config.hpp"
#include "app_data.hpp"
#include "host/host_dispatcher.hpp"
#include "redwood/backends.hpp"

// forward declare
void run_vulkan_demo();

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

#include "redwood/vulkan/engine.hpp"
#include "vulkan/vk_dispatcher.hpp"

void run_vulkan_demo() {
  vulkan::Engine engine;
  AppData app_data(engine.get_mr());

  vulkan::run_stage1(engine, app_data);
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

  spdlog::info("Done");
  return EXIT_SUCCESS;
}
