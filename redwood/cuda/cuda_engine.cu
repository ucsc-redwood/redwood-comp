// #include "../utils.hpp"
// #include "cuda_engine.cuh"
// #include "helpers.cuh"

// namespace cuda {

// Engine::Engine(bool manage_resources, const size_t num_streams)
//     : manage_resources_(manage_resources), streams_(num_streams) {
//   SPD_TRACE_FUNC;

//   streams_.resize(num_streams);
//   for (size_t i = 0; i < num_streams; ++i) {
//     CUDA_CHECK(cudaStreamCreate(&streams_[i]));
//   }
// }

// }  // namespace cuda
