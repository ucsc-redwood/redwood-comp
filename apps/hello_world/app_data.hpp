// #pragma once

// // #include "redwood/cuda/cuda_engine.cuh"
// // #include "redwood/uni_engine.hpp"
// // using BufferPtr = std::shared_ptr<cuda::TypedBuffer<T>>;

// #include <type_traits>

// // template <typename EngineT>
// // requires std::is_same_v<EngineT, cudaEngine> ||
// // struct AppData {
// //   explicit AppData(EngineT &eng, const size_t n) : n(n) {
// //     input_a = eng.buffer<int>(n)->fill(1);
// //     input_b = eng.buffer<int>(n)->fill(2);
// //     output = eng.buffer<int>(n)->zeros();
// //   }

// //   const size_t n;
// //   BufferPtr<int> input_a;
// //   BufferPtr<int> input_b;
// //   BufferPtr<int> output;
// // };
