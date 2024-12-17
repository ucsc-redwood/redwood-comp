#include "csr.hpp"

#include <cstdlib>

namespace v1 {
CSRMatrix denseToCsr(const float* dense_data,
                     const int rows,
                     const int cols,
                     const float threshold) {
  CSRMatrix sparse;
  sparse.rows = rows;
  sparse.cols = cols;
  sparse.row_ptr.push_back(0);

  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      float value = dense_data[row * cols + col];
      if (std::abs(value) > threshold) {
        sparse.values.push_back(value);
        sparse.col_indices.push_back(col);
        sparse.value_index_map[row * cols + col] = sparse.values.size() - 1;
      }
    }
    sparse.row_ptr.push_back(sparse.values.size());
  }
  return sparse;
}

}  // namespace v1

namespace v2 {

CSRMatrix::CSRMatrix(std::pmr::memory_resource* mr, int r, int c, int n)
    : BaseAppData(mr),
      rows(r),
      cols(c),
      nnz(n),
      u_values(n, mr),
      u_col_indices(n, mr),
      u_row_ptr(r + 1, mr) {}

}  // namespace v2
