#include <spdlog/spdlog.h>

#include "01_morton.cuh"
#include "02_sort.cuh"
#include "cu_dispatcher.cuh"
#include "im_storage.cuh"
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

// ----------------------------------------------------------------------------
// Stage 2 (sort) (morton code -> sorted morton code)
// ----------------------------------------------------------------------------

void run_stage2(AppData &app_data,
                ImStorage &im_storage,
                const cudaStream_t stream) {
  const auto n = app_data.get_n_input();

  const auto smem = 0;
  constexpr auto grid_size = 16;

  kernels::k_GlobalHistogram<<<grid_size,
                               ImStorage::GLOBAL_HIST_THREADS,
                               smem,
                               stream>>>(app_data.u_morton_keys.data(),
                                         im_storage.d_global_histogram,
                                         app_data.get_n_input());

  kernels::k_Scan<<<ImStorage::RADIX_PASSES, ImStorage::RADIX, 0, stream>>>(
      im_storage.d_global_histogram,
      im_storage.d_first_pass_histogram,
      im_storage.d_second_pass_histogram,
      im_storage.d_third_pass_histogram,
      im_storage.d_fourth_pass_histogram);

  kernels::k_DigitBinningPass<<<grid_size,
                                ImStorage::BINNING_THREADS,
                                0,
                                stream>>>(
      app_data.u_morton_keys.data(),  // <---
      app_data.u_morton_keys_alt.data(),
      im_storage.d_first_pass_histogram,
      im_storage.d_index,
      n,
      0);

  kernels::
      k_DigitBinningPass<<<grid_size, ImStorage::BINNING_THREADS, 0, stream>>>(
          app_data.u_morton_keys_alt.data(),
          app_data.u_morton_keys.data(),  // <---
          im_storage.d_second_pass_histogram,
          im_storage.d_index,
          n,
          8);

  kernels::k_DigitBinningPass<<<grid_size,
                                ImStorage::BINNING_THREADS,
                                0,
                                stream>>>(
      app_data.u_morton_keys.data(),  // <---
      app_data.u_morton_keys_alt.data(),
      im_storage.d_third_pass_histogram,
      im_storage.d_index,
      n,
      16);

  kernels::
      k_DigitBinningPass<<<grid_size, ImStorage::BINNING_THREADS, 0, stream>>>(
          app_data.u_morton_keys_alt.data(),
          app_data.u_morton_keys.data(),  // <---
          im_storage.d_fourth_pass_histogram,
          im_storage.d_index,
          n,
          24);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace cuda
