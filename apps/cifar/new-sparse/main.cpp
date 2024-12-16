#include "../../app.hpp"
// #include "../arg_max.hpp"
#include "app_data.hpp"

void run_cpu_demo_v1() {
  auto mr = std::pmr::new_delete_resource();
  AppData app_data(mr);

//   print_prediction(arg_max(app_data.u_linear_out.data()));
}

int main(int argc, char** argv) {
  INIT_APP("cifar-sparse");

  run_cpu_demo_v1();

  return 0;
}
