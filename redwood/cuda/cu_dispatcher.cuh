#include <cuda_runtime.h>

#include "../dispatcher.hpp"

namespace cuda {

class CuDispatcher final : public Dispatcher {
 public:
  CuDispatcher(std::shared_ptr<std::pmr::memory_resource> mr,
               const size_t n_concurrent);

  ~CuDispatcher();

  // Note: the dispatch_fn is not the kernel, it could be an lambda that
  // launches the kernel.
  void dispatch(const size_t queue_idx,
                std::function<void(size_t)> dispatch_fn) override;

  void synchronize(size_t queue_idx) override;

  // Cuda Specific
  [[nodiscard]] cudaStream_t stream(size_t queue_idx) const {
    return streams_[queue_idx];
  }

 private:
  std::vector<cudaStream_t> streams_;
};

}  // namespace cuda
