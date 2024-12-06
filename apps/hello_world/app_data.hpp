#pragma once

#include <algorithm>
#include <memory_resource>

template <typename T>
using UsmVector = std::pmr::vector<T>;

// App data(CPU)
struct AppData {
  explicit AppData(const size_t n, std::pmr::memory_resource* mr)
      : u_input_a(n, mr), u_input_b(n, mr), u_output(n, mr), n(n) {
    std::ranges::fill(u_input_a, 1.0f);
    std::ranges::fill(u_input_b, 2.0f);
    std::ranges::fill(u_output, 0.0f);
  }

  UsmVector<float> u_input_a;
  UsmVector<float> u_input_b;
  UsmVector<float> u_output;
  const size_t n;
};

// #ifdef REDWOOD_VULKAN_BACKEND

// #include "redwood/vulkan/vk_allocator.hpp"

// struct VulkanAppData {
//   explicit VulkanAppData(const size_t n, vulkan::VulkanMemoryResource& vk_mr)
//       : n(n) {
//     u_input_a = UsmVector<float>(n, &vk_mr);

//     vulkan::VulkanMemoryResource vk_mr1(vk_engine);
//     u_input_b = UsmVector<float>(n, &vk_mr1);

//     vulkan::VulkanMemoryResource vk_mr2(vk_engine);
//     u_output = UsmVector<float>(n, &vk_mr2);

//     std::ranges::fill(u_input_a, 1.0f);
//     std::ranges::fill(u_input_b, 2.0f);
//     std::ranges::fill(u_output, 0.0f);
//   }

//   UsmVector<float> u_input_a;
//   UsmVector<float> u_input_b;
//   UsmVector<float> u_output;
//   const size_t n;
// };

// #endif
