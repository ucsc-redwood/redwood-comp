#pragma once

#include <functional>

// a dispatcher is a singleton that allows you to dispatch any kernel to any
// data, and provide synchronization points.
// it also carries the memory resource for the application.
// or vulkan::Engine() in Vulkan backend.

class BaseDispatcher {
 public:
  explicit BaseDispatcher(const size_t n_concurrent)
      : n_concurrent_(n_concurrent) {}

  // takes a lambda that takes a queue_idx and performs the dispatch.
  virtual void dispatch(const size_t queue_idx,
                        std::function<void(size_t)> dispatch_fn,
                        bool sync = false) = 0;

  virtual void sync(size_t queue_idx) = 0;

 protected:
  const size_t n_concurrent_;
};
