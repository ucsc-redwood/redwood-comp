#pragma once

#include <cstddef>

// #include "redwood/base_appdata.hpp"

namespace cuda {

// This is device memory only, thus not using 'UsmVector'
struct ImStorage {
  ImStorage(int n);
  ~ImStorage();

  void clearSmem();

  static constexpr auto RADIX = 256;
  static constexpr auto RADIX_PASSES = 4;
  static constexpr auto BIN_PART_SIZE = 7680;
  static constexpr auto GLOBAL_HIST_THREADS = 128;
  static constexpr auto BINNING_THREADS = 512;

  size_t binning_blocks = 0;

  //   struct {
  unsigned int* d_global_histogram = nullptr;
  unsigned int* d_index = nullptr;
  unsigned int* d_first_pass_histogram = nullptr;
  unsigned int* d_second_pass_histogram = nullptr;
  unsigned int* d_third_pass_histogram = nullptr;
  unsigned int* d_fourth_pass_histogram = nullptr;
  int* u_flag_heads = nullptr;

  // UsmVector<unsigned int> d_global_histogram;
  // UsmVector<unsigned int> d_index;
  // UsmVector<unsigned int> d_first_pass_histogram;
  // UsmVector<unsigned int> d_second_pass_histogram;
  // UsmVector<unsigned int> d_third_pass_histogram;
  // UsmVector<unsigned int> d_fourth_pass_histogram;
  // UsmVector<int> u_flag_heads;

  //   } im_storage;
};

}  // namespace cuda
