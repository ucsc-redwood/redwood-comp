#include <spdlog/spdlog.h>

#include "01_morton.cuh"
#include "cu_dispatcher.cuh"
#include "redwood/cuda/helpers.cuh"

namespace cuda {

// ----------------------------------------------------------------------------
// Stage 1 (input -> morton code)
// ----------------------------------------------------------------------------

void run_stage1(AppData &app_data, const cudaStream_t stream) {
  static constexpr auto block_size = 256;
  const auto grid_size = div_up(app_data.get_n_input(), block_size);
  constexpr auto s_mem = 0;

  spdlog::debug(
      "CUDA kernel 'compute_morton_code', n = {}, threads = {}, blocks = {}, "
      "stream: {}",
      app_data.get_n_input(),
      block_size,
      grid_size,
      reinterpret_cast<void *>(stream));

  kernels::k_ComputeMortonCode<<<grid_size, block_size, s_mem, stream>>>(
      app_data.u_input_points.data(),
      app_data.u_morton_keys.data(),
      app_data.get_n_input(),
      app_data.min_coord,
      app_data.range);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace cuda
