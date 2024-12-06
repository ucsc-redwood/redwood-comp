#pragma once

#include <stdexcept>
#include <string>

// Helper function to handle CUDA errors
inline void cudaCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA Error: ") +
                             cudaGetErrorString(err) + " at " + file + ":" +
                             std::to_string(line));
  }
}

#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)
