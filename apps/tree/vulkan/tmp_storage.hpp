#pragma once

#include "redwood/base_appdata.hpp"

namespace vulkan {

struct TmpStorage : BaseAppData {
  TmpStorage(std::pmr::memory_resource* mr, const size_t n_input)
      : BaseAppData(mr), u_contributes(n_input, mr), u_out_idx(n_input, mr) {}

  UsmVector<uint32_t> u_contributes;
  UsmVector<uint32_t> u_out_idx;
};

}  // namespace vulkan