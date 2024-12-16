#include "../../app.hpp"
#include "app_data.hpp"
#include "host/host_kernels.hpp"

// [[nodiscard]] inline int arg_max(const float* ptr) {
//   const auto max_index =
//       std::distance(ptr, std::ranges::max_element(ptr, ptr + 10));

//   return max_index;
// }

// inline void print_prediction(const int max_index) {
//   static const std::unordered_map<int, std::string_view> class_names{
//       {0, "airplanes"},
//       {1, "cars"},
//       {2, "birds"},
//       {3, "cats"},
//       {4, "deer"},
//       {5, "dogs"},
//       {6, "frogs"},
//       {7, "horses"},
//       {8, "ships"},
//       {9, "trucks"}};

//   std::cout << "Predicted Image: ";
//   std::cout << (class_names.contains(max_index) ? class_names.at(max_index)
//                                                 : "Unknown");
//   std::cout << std::endl;
// }

void run_cpu_demo_v1() {
  auto mr = std::pmr::new_delete_resource();
  AppData app_data(mr);

  // clang-format off
  // conv2d_sparse(image_data, 3, 32, 32, conv1_weights, conv1_bias, 64, 3, 1, 1, true, conv1_output);
  // maxpool2d(conv1_output, 64, 32, 32, 2, 2, pool1_output);
  // conv2d_sparse(pool1_output, 64, 16, 16, conv2_weights, conv2_bias, 192, 3, 1, 1, true, conv2_output);
  // maxpool2d(conv2_output, 192, 16, 16, 2, 2, pool2_output);
  // conv2d_sparse(pool2_output, 192, 8, 8, conv3_weights, conv3_bias, 384, 3, 1, 1, true, conv3_output);
  // conv2d_sparse(conv3_output, 384, 8, 8, conv4_weights, conv4_bias, 256, 3, 1, 1, true, conv4_output);
  // conv2d_sparse(conv4_output, 256, 8, 8, conv5_weights, conv5_bias, 256, 3, 1, 1, true, conv5_output);
  // maxpool2d(conv5_output, 256, 8, 8, 2, 2, pool3_output);
  // linearLayer_sparse(pool3_output, linear_weights, linear_bias, linear_output);
  // clang-format on

  cpu::kernels::sparse::conv2d(app_data.u_image_data.data(),
                               3,
                               32,
                               32,
                               app_data.conv1_weights,
                               app_data.u_conv1_bias.data(),
                               64,
                               3,
                               1,
                               1,
                               true,
                               app_data.u_conv1_output.data());

  cpu::kernels::sparse::maxpool2d(app_data.u_conv1_output.data(),
                                  64,
                                  32,
                                  32,
                                  2,
                                  2,
                                  app_data.u_pool1_output.data());

  cpu::kernels::sparse::conv2d(app_data.u_pool1_output.data(),
                               64,
                               16,
                               16,
                               app_data.conv2_weights,
                               app_data.u_conv2_bias.data(),
                               192,
                               3,
                               1,
                               1,
                               true,
                               app_data.u_conv2_output.data());

  cpu::kernels::sparse::maxpool2d(app_data.u_conv2_output.data(),
                                  192,
                                  16,
                                  16,
                                  2,
                                  2,
                                  app_data.u_pool2_output.data());

  cpu::kernels::sparse::conv2d(app_data.u_pool2_output.data(),
                               192,
                               8,
                               8,
                               app_data.conv3_weights,
                               app_data.u_conv3_bias.data(),
                               384,
                               3,
                               1,
                               1,
                               true,
                               app_data.u_conv3_output.data());

  cpu::kernels::sparse::conv2d(app_data.u_conv3_output.data(),
                               384,
                               8,
                               8,
                               app_data.conv4_weights,
                               app_data.u_conv4_bias.data(),
                               256,
                               3,
                               1,
                               1,
                               true,
                               app_data.u_conv4_output.data());

  cpu::kernels::sparse::conv2d(app_data.u_conv4_output.data(),
                               256,
                               8,
                               8,
                               app_data.conv5_weights,
                               app_data.u_conv5_bias.data(),
                               256,
                               3,
                               1,
                               1,
                               true,
                               app_data.u_conv5_output.data());

  cpu::kernels::sparse::maxpool2d(app_data.u_conv5_output.data(),
                                  256,
                                  8,
                                  8,
                                  2,
                                  2,
                                  app_data.u_pool3_output.data());

  cpu::kernels::sparse::linear(app_data.u_pool3_output.data(),
                               app_data.linear_weights,
                               app_data.u_linear_bias.data(),
                               app_data.u_linear_output.data());

  // print_prediction(arg_max(app_data.u_linear_output.data()));

  // Find the index of the maximum element in the linear layer output
  int max_index = 0;
  float max_value = app_data.u_linear_output.data()[0];
  for (int i = 1; i < 10; ++i) {
    if (app_data.u_linear_output.data()[i] > max_value) {
      max_value = app_data.u_linear_output.data()[i];
      max_index = i;
    }
  }

  // Map the index to the corresponding class and print the prediction
  std::cout << "Predicted Image: ";
  switch (max_index) {
    case 0:
      std::cout << "airplanes";
      break;
    case 1:
      std::cout << "cars";
      break;
    case 2:
      std::cout << "birds";
      break;
    case 3:
      std::cout << "cats";
      break;
    case 4:
      std::cout << "deer";
      break;
    case 5:
      std::cout << "dogs";
      break;
    case 6:
      std::cout << "frogs";
      break;
    case 7:
      std::cout << "horses";
      break;
    case 8:
      std::cout << "ships";
      break;
    case 9:
      std::cout << "trucks";
      break;
    default:
      std::cout << "Unknown";
      break;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  INIT_APP("cifar-sparse");

  run_cpu_demo_v1();

  return 0;
}
