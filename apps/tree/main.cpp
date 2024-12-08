#include <spdlog/spdlog.h>

#include "../cli_to_config.hpp"
#include "../read_config.hpp"
#include "app_data.hpp"
#include "host/host_dispatchers.hpp"

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
  cpu::run_stage3(app_data, pool, n_threads);
  cpu::run_stage4(app_data, pool, n_threads);
  cpu::run_stage5(app_data, pool, n_threads);
  cpu::run_stage6(app_data, pool, n_threads);
  cpu::run_stage7(app_data, pool, n_threads);

  print_stats(app_data);
}

int main(int argc, char** argv) {
  auto config = helpers::init_demo(argc, argv);
  auto small_cores = helpers::get_cores_by_type(config["cpu_info"], "small");
  auto medium_cores = helpers::get_cores_by_type(config["cpu_info"], "medium");
  auto big_cores = helpers::get_cores_by_type(config["cpu_info"], "big");

  assert(!small_cores.empty());

  spdlog::set_level(spdlog::level::trace);

  run_cpu_demo(small_cores, 640 * 480);

  spdlog::info("Done.");
  return 0;
}
