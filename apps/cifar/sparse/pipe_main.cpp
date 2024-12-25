#include "../../app.hpp"
#include "app_data.hpp"
#include "host/host_dispatcher.hpp"
// #include "redwood/backends.hpp"

constexpr auto n_iterations = 10000;



void run_cpu_2_stage() {
  auto mr = std::pmr::new_delete_resource();
  
  AppData app_data_1(mr);
  AppData app_data_2(mr);
  
  std::vector<int> pool_1_cores_ids = {0, 1, 2, 3};
  std::vector<int> pool_2_cores_ids = {4, 5, 6, 7};

  core::thread_pool pool_1(pool_1_cores_ids, true);
  core::thread_pool pool_2(pool_2_cores_ids, true);


  


}


int main(int argc, char **argv) {
  INIT_APP("pipe-cifar-sparse");

  spdlog::info("Running CPU 2 stage");
  run_cpu_2_stage();

  return EXIT_SUCCESS;
}
