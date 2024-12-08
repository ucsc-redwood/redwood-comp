#include "im_storage.cuh"
#include "redwood/cuda/helpers.cuh"

namespace cuda {

template <typename T>
constexpr void malloc_managed(T** ptr, const size_t num_items) {
  CUDA_CHECK(
      cudaMallocManaged(reinterpret_cast<void**>(ptr), num_items * sizeof(T)));
}

template <typename T>
constexpr void malloc_device(T** ptr, const size_t num_items) {
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), num_items * sizeof(T)));
}

#define MALLOC_MANAGED(ptr, num_items) malloc_managed(ptr, num_items)

#define MALLOC_DEVICE(ptr, num_items) malloc_device(ptr, num_items)

#define CUDA_FREE(ptr) CUDA_CHECK(cudaFree(ptr))

#define SET_MEM_2_ZERO(ptr, item_count) \
  CUDA_CHECK(cudaMemsetAsync(           \
      ptr, 0, sizeof(std::remove_pointer_t<decltype(ptr)>) * (item_count)))

ImStorage::ImStorage(const int n) {
  // binning_blocks = 270 usually (w/ ~2M points)
  MALLOC_DEVICE(&d_global_histogram, RADIX * RADIX_PASSES);
  MALLOC_DEVICE(&d_index, RADIX_PASSES);
  MALLOC_DEVICE(&d_first_pass_histogram, RADIX * binning_blocks);
  MALLOC_DEVICE(&d_second_pass_histogram, RADIX * binning_blocks);
  MALLOC_DEVICE(&d_third_pass_histogram, RADIX * binning_blocks);
  MALLOC_DEVICE(&d_fourth_pass_histogram, RADIX * binning_blocks);

  MALLOC_MANAGED(&u_flag_heads, n);
}

ImStorage::~ImStorage() {
  CUDA_FREE(d_global_histogram);
  CUDA_FREE(d_index);
  CUDA_FREE(d_first_pass_histogram);
  CUDA_FREE(d_second_pass_histogram);
  CUDA_FREE(d_third_pass_histogram);
  CUDA_FREE(d_fourth_pass_histogram);

  CUDA_FREE(u_flag_heads);
}

void ImStorage::clearSmem() {
  SET_MEM_2_ZERO(d_global_histogram, RADIX * RADIX_PASSES);
  SET_MEM_2_ZERO(d_index, RADIX_PASSES);
  SET_MEM_2_ZERO(d_first_pass_histogram, RADIX * binning_blocks);
  SET_MEM_2_ZERO(d_second_pass_histogram, RADIX * binning_blocks);
  SET_MEM_2_ZERO(d_third_pass_histogram, RADIX * binning_blocks);
  SET_MEM_2_ZERO(d_fourth_pass_histogram, RADIX * binning_blocks);
}

}  // namespace cuda
