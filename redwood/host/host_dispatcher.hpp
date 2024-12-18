#pragma once

#include "../base_dispatcher.hpp"
#include "thread_pool.hpp"

class HostDispatcher : public BaseDispatcher {
 public:
  // Unpinned default thread pool
  HostDispatcher(const size_t n_concurrent) : BaseDispatcher(n_concurrent) {}

  // Pinned specific cores
  HostDispatcher(const size_t n_concurrent, std::vector<int> core_ids)
      : BaseDispatcher(n_concurrent) {
    core_ids.reserve(n_concurrent);
  }

 private:
  std::vector<core::thread_pool> pools_;
};
