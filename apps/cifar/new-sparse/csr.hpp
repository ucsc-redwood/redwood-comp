#pragma once

// #include "redwood/base_appdata.hpp"

// note this pointer may came from USM vector
struct CSRMatrix {
    const float* values;
    const int* row_ptr;
    const int* col_idx;
    int rows;
    int cols;
    int nnz;
};

// struct CSRMatrix : public BaseAppData {
//   explicit CSRMatrix(std::pmr::memory_resource* mr)
//       : BaseAppData(mr), values(mr), row_ptr(mr), col_idx(mr) {
//       }

//   UsmVector<float> values;
//   UsmVector<int> row_ptr;
//   UsmVector<int> col_idx;
//   int rows;
//   int cols;
//   int nnz;
// };
