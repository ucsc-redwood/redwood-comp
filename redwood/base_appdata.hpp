#pragma once

#include <memory_resource>
#include <vector>

template <typename T>
using UsmVector = std::pmr::vector<T>;

// ----------------------------------------------------------------------------
// BaseAppData interface
// For all demo applications, you want to create a class that inherits from
// BaseAppData. This gives you a common interface for all backends.
// ----------------------------------------------------------------------------

struct BaseAppData {
  explicit BaseAppData(std::pmr::memory_resource* mr) : mr_(mr) {}

  virtual ~BaseAppData() = default;

  std::pmr::memory_resource* mr_;
};
