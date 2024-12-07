#include <iostream>

#include "base_engine.hpp"
#include "vma_pmr.hpp"

int main() {
  spdlog::set_level(spdlog::level::trace);

  BaseEngine engine;

  VulkanMemoryResource mr(engine.get_device());

  const size_t n = 1024;
  std::pmr::vector<int> vec1(n, &mr);
  std::pmr::vector<int> vec2(n, &mr);

  std::cout << "vec1 size address: " << (void *)vec1.data() << std::endl;
  std::cout << "vec2 size address: " << (void *)vec2.data() << std::endl;

  std::ranges::fill(vec1, 1);
  std::ranges::fill(vec2, 2);

  vk::Buffer vec1_buffer = mr.get_buffer_from_pointer(vec1.data());

  return 0;
}
