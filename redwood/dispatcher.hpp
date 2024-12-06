#pragma once

#include <functional>
#include <memory>

// a dispatcher is a singleton that allows you to dispatch any kernel to any
// data, and provide synchronization points.
// it also carries the memory resource for the application.
// or vulkan::Engine() in Vulkan backend.

class Dispatcher {
 public:
  explicit Dispatcher(std::shared_ptr<std::pmr::memory_resource> mr,
                      const size_t n_concurrent)
      : n_concurrent_(n_concurrent), mr_(std::move(mr)) {}

  // takes a lambda that takes a queue_idx and performs the dispatch.
  virtual void dispatch(const size_t queue_idx,
                        std::function<void(size_t)> dispatch_fn) = 0;

  virtual void synchronize(size_t queue_idx) = 0;

 protected:
  const size_t n_concurrent_;
  std::shared_ptr<std::pmr::memory_resource> mr_;
};
