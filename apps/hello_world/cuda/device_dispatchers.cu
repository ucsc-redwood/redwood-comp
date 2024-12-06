#include <spdlog/spdlog.h>

#include "device_dispatchers.cuh"
#include "device_kernels.cuh"
#include "redwood/cuda/helpers.cuh"

namespace cuda {

void run_stage1(AppData &app_data, const cudaStream_t stream) {
  constexpr auto threads = 256;
  const auto blocks = div_up(app_data.n, threads);
  constexpr auto s_mem = 0;

  spdlog::debug(
      "CUDA kernel 'vector_add', n = {}, threads = {}, blocks = {}, stream: {}",
      app_data.n,
      threads,
      blocks,
      reinterpret_cast<void *>(stream));

  cuda::kernels::vector_add<<<blocks, threads, s_mem, stream>>>(
      app_data.u_input_a.data(),
      app_data.u_input_b.data(),
      app_data.u_output.data(),
      app_data.n);
}

}  // namespace cuda
