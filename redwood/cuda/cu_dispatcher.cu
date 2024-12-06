#include "../utils.hpp"
#include "cu_dispatcher.cuh"
#include "helpers.cuh"

namespace cuda {

CuDispatcher::CuDispatcher(std::shared_ptr<std::pmr::memory_resource> mr,
                           const size_t n_concurrent)
    : Dispatcher(std::move(mr), n_concurrent), streams_(n_concurrent) {
  SPD_TRACE_FUNC;

  for (auto& stream : streams_) {
    CUDA_CHECK(cudaStreamCreate(&stream));
  }
}

CuDispatcher::~CuDispatcher() {
  SPD_TRACE_FUNC;

  for (auto& stream : streams_) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
}

void CuDispatcher::dispatch(const size_t queue_idx,
                            std::function<void(size_t)> dispatch_fn) {
  SPD_TRACE_FUNC;

  dispatch_fn(queue_idx);
}

void CuDispatcher::sync(size_t queue_idx) {
  SPD_TRACE_FUNC;

  CUDA_CHECK(cudaStreamSynchronize(streams_[queue_idx]));
}

}  // namespace cuda
