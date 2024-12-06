#include <spdlog/spdlog.h>

#include "device_dispatchers.cuh"
#include "device_kernels.cuh"
#include "redwood/cuda/helpers.cuh"

namespace cuda {

void run_stage1(Engine &engine, const cuda::AppData &app_data) {
  constexpr dim3 threads_per_block(256);
  const dim3 blocks(div_up(app_data.n, threads_per_block.x));
  constexpr auto s_mem = 0;
  const auto stream = engine.stream(0);

  spdlog::debug(
      "CUDA kernel 'vector_add', n = {}, threads_per_block = {}, "
      "blocks = {}, on stream: {}",
      app_data.n,
      threads_per_block.x,
      blocks.x,
      (void *)stream);

  cuda::kernels::vector_add<<<blocks, threads_per_block, s_mem, stream>>>(
      app_data.input_a->data(),
      app_data.input_b->data(),
      app_data.output->data(),
      0,
      app_data.n);

  spdlog::debug("Synchronizing CUDA stream: {}", (void *)stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace cuda
