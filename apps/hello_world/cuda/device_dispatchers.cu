// #include <spdlog/spdlog.h>

// #include "device_dispatchers.cuh"
// #include "device_kernels.cuh"
// #include "redwood/cuda/helpers.cuh"

// namespace cuda {

// void run_stage1(cuda::Engine &engine, const AppData &app_data) {
//   constexpr auto threads = 256;
//   const auto blocks = div_up(app_data.n, threads);
//   constexpr auto s_mem = 0;
//   const auto stream = engine.stream(0);

//   spdlog::debug(
//       "CUDA kernel 'vector_add', n = {}, threads = {}, blocks = {}, on
//       stream: "
//       "{}",
//       app_data.n,
//       threads,
//       blocks,
//       reinterpret_cast<void *>(stream));

//   cuda::kernels::vector_add<<<blocks, threads, s_mem, stream>>>(
//       app_data.input_a->data(),
//       app_data.input_b->data(),
//       app_data.output->data(),
//       0,
//       app_data.n);
// }

// }  // namespace cuda
