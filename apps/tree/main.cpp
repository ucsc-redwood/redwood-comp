#include <spdlog/spdlog.h>

#include "../cli_to_config.hpp"
#include "../read_config.hpp"
#include "app_data.hpp"
#include "host/host_dispatchers.hpp"

int main(int argc, char** argv) {
  auto config = helpers::init_demo(argc, argv);
  auto small_cores = helpers::get_cores_by_type(config["cpu_info"], "small");
  auto medium_cores = helpers::get_cores_by_type(config["cpu_info"], "medium");
  auto big_cores = helpers::get_cores_by_type(config["cpu_info"], "big");

  assert(!small_cores.empty());

  spdlog::set_level(spdlog::level::trace);

  constexpr auto n = 640 * 480;  // = 307200

  AppData app_data(std::pmr::new_delete_resource(), n);

  core::thread_pool pool(small_cores);
  cpu::run_stage1(app_data, pool, small_cores.size());
  cpu::run_stage2(app_data, pool, small_cores.size());
  cpu::run_stage3(app_data, pool, small_cores.size());
  cpu::run_stage4(app_data, pool, small_cores.size());

  // print the first 10 sorted morton keys
  for (auto i = 0; i < 10; ++i) {
    spdlog::info(
        "u_morton_keys[{}] = {}", i, app_data.get_unique_morton_keys()[i]);
  }

  if (std::is_sorted(
          app_data.get_unique_morton_keys(),
          app_data.get_unique_morton_keys() + app_data.get_n_unique())) {
    spdlog::info("u_morton_keys is sorted.");
  } else {
    spdlog::error("u_morton_keys is not sorted!");
  }

  spdlog::info("Done.");
  return 0;
}
