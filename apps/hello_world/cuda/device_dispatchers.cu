#include <spdlog/spdlog.h>

#include "device_dispatchers.cuh"
#include "device_kernels.cuh"
#include "redwood/cuda/helpers.cuh"

namespace cuda {

void run_stage1(AppData &app_data) {
  constexpr auto threads = 256;
  const auto blocks = div_up(app_data.n, threads);
  constexpr auto s_mem = 0;
  //   const auto stream = engine.stream(0);

  spdlog::debug(
      "CUDA kernel 'vector_add', n = {}, threads = {}, blocks = {}, on",
      //   stream: "
      //   "{}",
      app_data.n,
      threads,
      blocks
      //   reinterpret_cast<void *>(stream)
  );

  cuda::kernels::vector_add<<<blocks, threads, s_mem>>>(
      app_data.u_input_a.data(),
      app_data.u_input_b.data(),
      app_data.u_output.data(),
      0,
      app_data.n);

  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace cuda
