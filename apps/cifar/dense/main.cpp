#include "../../cli_to_config.hpp"
#include "../../read_config.hpp"
#include "app_data.hpp"
#include "host/host_dispatcher.hpp"

void run_cpu_demo(const std::vector<int>& small_cores, const size_t n_threads) {
  auto mr = std::pmr::new_delete_resource();
  AppData app_data(mr);

  core::thread_pool pool(small_cores);
  cpu::run_stage1(app_data, pool, n_threads).wait();
}

int main(int argc, char** argv) {
  auto config = helpers::init_demo(argc, argv);
  auto small_cores = helpers::get_cores_by_type(config["cpu_info"], "small");
  auto medium_cores = helpers::get_cores_by_type(config["cpu_info"], "medium");
  auto big_cores = helpers::get_cores_by_type(config["cpu_info"], "big");

  assert(!small_cores.empty());

  spdlog::info("Small cores: [{}]", fmt::join(small_cores, ", "));
  spdlog::info("Medium cores: [{}]", fmt::join(medium_cores, ", "));
  spdlog::info("Big cores: [{}]", fmt::join(big_cores, ", "));

  spdlog::set_level(spdlog::level::trace);

  run_cpu_demo(small_cores, small_cores.size());

  spdlog::info("Done");
  return EXIT_SUCCESS;
}
