// #pragma once

// #include "buffer.hpp"
// #include <memory>
// #include <vector>

// enum class EngineType {
//   kCpu,
//   kCuda,
//   kVulkan,
// };

// class Engine {
// public:
//   Engine(EngineType type);

//   virtual std::shared_ptr<BaseBuffer> buffer(size_t size) = 0;

// private:
//   const EngineType type_;

//   std::vector<std::weak_ptr<BaseBuffer>> buffers_;
// };
