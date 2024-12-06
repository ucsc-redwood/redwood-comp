// #pragma once

// #include <memory>
// #include <vector>

// #include "base_buffer.hpp"

// class UniEngine {
//  public:
//   UniEngine(const bool manage_resources = true, const size_t num_streams = 4)
//       : manage_resources_(manage_resources), num_streams_(num_streams) {
//     spdlog::info(
//         "Creating UniEngine with {} streams and {} resource management",
//         num_streams_,
//         manage_resources_ ? "enabled" : "disabled");
//   }

//   template <typename BufferT>
//     requires std::is_base_of_v<BaseBuffer, BufferT>
//   [[nodiscard]] std::shared_ptr<BufferT> buffer(const size_t size) {
//     const auto buffer = std::make_shared<BufferT>(size);
//     if (manage_resources_) {
//       buffers_.push_back(buffer);
//     }
//     return buffer;
//   }

//   std::vector<std::weak_ptr<BaseBuffer>> buffers_;
//   const bool manage_resources_;
//   const size_t num_streams_;
// };
