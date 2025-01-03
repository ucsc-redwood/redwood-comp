#pragma once

#include <unordered_map>
#include <vector>

#include "redwood/base_appdata.hpp"
// ----------------------------------------------------------------------------
// CSR Format (Old)
// ----------------------------------------------------------------------------

namespace v1 {

// Optimized CSR format structure
struct CSRMatrix {
  std::vector<float> values;
  std::vector<int> col_indices;
  std::vector<int> row_ptr;
  std::unordered_map<int, int> value_index_map;
  int rows;
  int cols;
};

// Optimized dense to CSR conversion
[[nodiscard, deprecated("Use the new CSR format instead")]] CSRMatrix
denseToCsr(const float* dense_data, int rows, int cols, float threshold = 1e-6);

}  // namespace v1

namespace v2 {

struct CSRMatrix final : public BaseAppData {
  explicit CSRMatrix(std::pmr::memory_resource* mr, int r, int c, int n);

  int rows{};
  int cols{};
  int nnz{};  // Number of non-zero entries

  // Device pointers
  //   double* d_values{nullptr};    // of length nnz
  //   int* d_col_indices{nullptr};  // of length nnz
  //   int* d_row_ptr{nullptr};      // of length rows+1

  UsmVector<float> u_values;
  UsmVector<int> u_col_indices;
  UsmVector<int> u_row_ptr;

};

}  // namespace v2


namespace v3 {

  

}