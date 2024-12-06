#pragma once

#include <algorithm>

#include "redwood/base_appdata.hpp"

struct AppData final : public BaseAppData {
  explicit AppData(std::pmr::memory_resource* mr, const size_t n)
      : BaseAppData(mr),
        u_input_a(n, mr),
        u_input_b(n, mr),
        u_output(n, mr),
        n(n) {
    std::ranges::fill(u_input_a, 1.0f);
    std::ranges::fill(u_input_b, 2.0f);
    std::ranges::fill(u_output, 0.0f);
  }

  ~AppData() override = default;

  UsmVector<float> u_input_a;
  UsmVector<float> u_input_b;
  UsmVector<float> u_output;
  const size_t n;
};
