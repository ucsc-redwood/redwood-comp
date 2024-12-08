#include <spdlog/spdlog.h>

#include "../cli_to_config.hpp"
#include "../read_config.hpp"
#include "app_data.hpp"

int main(int argc, char** argv) {
  auto config = helpers::init_demo(argc, argv);
  auto small_cores = helpers::get_cores_by_type(config["cpu_info"], "small");
  auto medium_cores = helpers::get_cores_by_type(config["cpu_info"], "medium");
  auto big_cores = helpers::get_cores_by_type(config["cpu_info"], "big");

  assert(!small_cores.empty());

  spdlog::set_level(spdlog::level::trace);

  constexpr auto n = 640 * 480;  // = 307200

  AppData app_data(std::pmr::new_delete_resource(), n);

  spdlog::info("Done.");
  return 0;
}
