#include "../app_data.hpp"

namespace model {
constexpr float sparse_threshold = 1e-6;
}

struct SparseAppData final : public AppData {
  explicit SparseAppData(std::pmr::memory_resource* mr);
};
