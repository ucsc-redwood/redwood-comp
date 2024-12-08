#pragma once

#include <barrier>

namespace cpu {

namespace kernels {

void k_binning_pass(const size_t tid,
                    std::barrier<>& barrier,
                    const uint32_t* u_sort_begin,
                    const uint32_t* u_sort_end,
                    uint32_t* u_sort_alt,  // output
                    const int shift);

// core::multi_future<void> dispatch_binning_pass(const int n_bins,
//                                                const int n_points,
//                                                const int *bin_indices,
//                                                const float *points,
//                                                const float *bin_min,
//                                                const float *bin_max,
//                                                int *bin_counts);

}  // namespace kernels

}  // namespace cpu
