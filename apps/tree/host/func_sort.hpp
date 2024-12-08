#pragma once

#include <barrier>

#include "redwood/host/thread_pool.hpp"

namespace cpu {

namespace kernels {

void k_binning_pass(const size_t tid,
                    std::barrier<>& barrier,
                    const uint32_t* u_sort_begin,
                    const uint32_t* u_sort_end,
                    uint32_t* u_sort_alt,  // output
                    const int shift);

core::multi_future<void> dispatch_binning_pass(core::thread_pool& pool,
                                               const size_t n_threads,
                                               std::barrier<>& barrier,
                                               const int n,
                                               const uint32_t* u_sort,
                                               uint32_t* u_sort_alt,
                                               const int shift);

}  // namespace kernels

}  // namespace cpu
