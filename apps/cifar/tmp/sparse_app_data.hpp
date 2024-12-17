#include "../app_data.hpp"
#include "csr.hpp"

namespace model {
constexpr float sparse_threshold = 1e-6;
}

struct SparseAppData final : public AppData {
  explicit SparseAppData(std::pmr::memory_resource* mr);

  // Precomputed CSR matrices
  v1::CSRMatrix sparse_image;
  v1::CSRMatrix sparse_conv1_weights;
  v1::CSRMatrix sparse_conv2_weights;
  v1::CSRMatrix sparse_conv3_weights;
  v1::CSRMatrix sparse_conv4_weights;
  v1::CSRMatrix sparse_conv5_weights;
  v1::CSRMatrix sparse_linear_weights;
};
