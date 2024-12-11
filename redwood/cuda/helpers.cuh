#pragma once

#include <stdexcept>
#include <string>

// ----------------------------------------------------------------------------
// Helper function to handle CUDA errors
// ----------------------------------------------------------------------------

inline void cudaCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA Error: ") +
                             cudaGetErrorString(err) + " at " + file + ":" +
                             std::to_string(line));
  }
}

#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// Simplify launch parameters
// Need to define TOTAL_ITER (e.g., 'total_iter' = 10000), and then write some
// number for BLOCK_SIZE (e.g., 256)
// ----------------------------------------------------------------------------

#define SETUP_DEFAULT_LAUNCH_PARAMS(TOTAL_ITER, BLOCK_SIZE)     \
  static constexpr auto block_dim = dim3{BLOCK_SIZE, 1, 1};     \
  static const auto grid_dim = div_up(TOTAL_ITER, block_dim.x); \
  static constexpr auto shared_mem = 0;

// ----------------------------------------------------------------------------
// Math
// ----------------------------------------------------------------------------

constexpr size_t div_up(const size_t a, const size_t b) {
  return (a + b - 1) / b;
}
