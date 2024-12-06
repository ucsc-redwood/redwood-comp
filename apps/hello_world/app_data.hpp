#pragma once

#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory_resource>

template <typename T>
using UsmVector = std::pmr::vector<T>;

struct AppData {
  explicit AppData(const size_t n, std::pmr::memory_resource* mr)
      : u_input_a(n, mr), u_input_b(n, mr), u_output(n, mr), n(n) {
    spdlog::debug("Allocated AppData with size {}", n);
    std::ranges::fill(u_input_a, 1.0f);
    std::ranges::fill(u_input_b, 2.0f);
    std::ranges::fill(u_output, 0.0f);
  }

  UsmVector<float> u_input_a;
  UsmVector<float> u_input_b;
  UsmVector<float> u_output;
  const size_t n;
};
