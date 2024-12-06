// #pragma once

// #include <cuda_runtime.h>

// #include <memory>
// #include <vector>

// #include "cu_buffer.cuh"
// // #include "cu_buffer_typed.cuh"
// #include "helpers.cuh"

// namespace cuda {

// class Engine {
//  public:
//   Engine(bool manage_resources = true, const size_t num_streams = 4);

//   // template <typename T, typename... Args>
//   //   requires std::is_constructible_v<cuda::TypedBuffer<T>, Args...>
//   // [[nodiscard]] std::shared_ptr<cuda::TypedBuffer<T>> buffer(Args
//   &&...args) {
//   //   const auto buffer = std::make_shared<cuda::TypedBuffer<T>>(args...);

//   //   if (manage_resources_) {
//   //     buffers_.push_back(buffer);
//   //   }

//   //   return buffer;
//   // }

//   [[nodiscard]] cudaStream_t stream(const size_t i) const {
//     return streams_[i];
//   }

//   void sync(const cudaStream_t stream) const {
//     spdlog::debug("Synchronizing CUDA stream: {}",
//                   reinterpret_cast<void *>(stream));
//     CUDA_CHECK(cudaStreamSynchronize(stream));
//   }

//   void sync() const {
//     for (const auto &stream : streams_) {
//       sync(stream);
//     }
//   }

//  private:
//   std::vector<std::weak_ptr<cuda::Buffer>> buffers_;
//   std::vector<cudaStream_t> streams_;

//   const bool manage_resources_ = true;
// };

// }  // namespace cuda
