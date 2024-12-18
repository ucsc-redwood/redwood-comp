#pragma once

#include "../base_dispatcher.hpp"
#include "algorithm.hpp"
#include "engine.hpp"
#include "redwood/vulkan/sequence.hpp"

namespace vulkan {

class VulkanDispatcher : public BaseDispatcher {
 public:
  VulkanDispatcher(const size_t n_concurrent, Engine &engine)
      : BaseDispatcher(n_concurrent), engine_ref(engine) {
    sequences_.reserve(n_concurrent);
    for (size_t i = 0; i < n_concurrent; ++i) {
      sequences_.push_back(engine.sequence());
    }
  }

 protected:
  Engine &engine_ref;
  std::unordered_map<std::string, std::shared_ptr<Algorithm>> algorithms;

  std::vector<std::shared_ptr<Sequence>> sequences_;
};

}  // namespace vulkan
