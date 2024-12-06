#include "device_dispatchers.cuh"
#include "device_kernels.cuh"

namespace cuda {

void run_stage1(const cuda::AppData &app_data) {
  // use 256 threads per block
  constexpr dim3 threads_per_block(256);
  const dim3 blocks((app_data.n + threads_per_block.x - 1) /
                    threads_per_block.x);

  cuda::kernels::vector_add<<<blocks, threads_per_block, 0>>>(
      app_data.input_a->data(), app_data.input_b->data(),
      app_data.output->data(), 0, app_data.n);

  cudaDeviceSynchronize();
}

} // namespace cuda
