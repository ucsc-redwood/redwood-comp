#pragma once

#include <cuda_runtime_api.h>

#include "../base_dispatcher.hpp"
#include "helpers.cuh"

namespace cuda {

class CudaDispatcher : public BaseDispatcher {
 public:
  CudaDispatcher(const size_t n_concurrent) : BaseDispatcher(n_concurrent) {
    streams_.reserve(n_concurrent);
    for (size_t i = 0; i < n_concurrent; ++i) {
      CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }
  }

  ~CudaDispatcher() {
    for (auto &stream : streams_) {
      CUDA_CHECK(cudaStreamDestroy(stream));
    }
  }

  void dispatch(const size_t queue_idx,
                std::function<void(size_t)> dispatch_fn,
                const bool sync = false) override {
    dispatch_fn(queue_idx);

    if (sync) {
      this->sync(queue_idx);
    }
  }

  void sync(const size_t queue_idx) override {
    CUDA_CHECK(cudaStreamSynchronize(streams_[queue_idx]));
  };

  std::vector<cudaStream_t> streams_;
};

}  // namespace cuda