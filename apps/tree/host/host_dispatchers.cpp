#include "host_dispatchers.hpp"

#include <spdlog/spdlog.h>

#include "func_brt.hpp"
#include "func_edge.hpp"
#include "func_morton.hpp"
#include "func_octree.hpp"
#include "func_sort.hpp"

namespace cpu {

// ----------------------------------------------------------------------------
// Stage 1 (xyz -> morton)
// ----------------------------------------------------------------------------

void run_stage1(AppData& app_data,
                core::thread_pool& pool,
                const size_t n_threads) {
  const int start = 0;
  const int end = app_data.get_n_input();

  constexpr float min_coord = 0.0f;
  constexpr float range = 1024.0f;

  // spd log, show kernel name, number of threads, number of items
  spdlog::debug("[cpu] run_stage1 xyz_to_morton32: {} threads, {} items",
                n_threads,
                app_data.get_n_input());

  pool.submit_blocks(
          start,
          end,
          [&](const int start, const int end) {
            for (int i = start; i < end; ++i) {
              app_data.u_morton_keys[i] = kernels::xyz_to_morton32(
                  app_data.u_input_points[i], min_coord, range);
            }
          },
          n_threads)
      .wait();
}

// ----------------------------------------------------------------------------
// Stage 2 (morton -> sorted morton)
// ----------------------------------------------------------------------------

template <typename T>
class [[nodiscard]] my_blocks {
 public:
  /**
   * @brief Construct a `blocks` object with the given specifications.
   *
   * @param first_index_ The first index in the range.
   * @param index_after_last_ The index after the last index in the range.
   * @param num_blocks_ The desired number of blocks to divide the range into.
   */
  my_blocks(const T first_index_,
            const T index_after_last_,
            const size_t num_blocks_)
      : first_index(first_index_),
        index_after_last(index_after_last_),
        num_blocks(num_blocks_) {
    if (index_after_last > first_index) {
      const size_t total_size =
          static_cast<size_t>(index_after_last - first_index);
      if (num_blocks > total_size) num_blocks = total_size;
      block_size = total_size / num_blocks;
      remainder = total_size % num_blocks;
      if (block_size == 0) {
        block_size = 1;
        num_blocks = (total_size > 1) ? total_size : 1;
      }
    } else {
      num_blocks = 0;
    }
  }

  /**
   * @brief Get the first index of a block.
   *
   * @param block The block number.
   * @return The first index.
   */
  [[nodiscard]] T start(const size_t block) const {
    return first_index + static_cast<T>(block * block_size) +
           static_cast<T>(block < remainder ? block : remainder);
  }

  /**
   * @brief Get the index after the last index of a block.
   *
   * @param block The block number.
   * @return The index after the last index.
   */
  [[nodiscard]] T end(const size_t block) const {
    return (block == num_blocks - 1) ? index_after_last : start(block + 1);
  }

  /**
   * @brief Get the number of blocks. Note that this may be different than the
   * desired number of blocks that was passed to the constructor.
   *
   * @return The number of blocks.
   */
  [[nodiscard]] size_t get_num_blocks() const { return num_blocks; }

 private:
  /**
   * @brief The size of each block (except possibly the last block).
   */
  size_t block_size = 0;

  /**
   * @brief The first index in the range.
   */
  T first_index = 0;

  /**
   * @brief The index after the last index in the range.
   */
  T index_after_last = 0;

  /**
   * @brief The number of blocks.
   */
  size_t num_blocks = 0;

  /**
   * @brief The remainder obtained after dividing the total size by the number
   * of blocks.
   */
  size_t remainder = 0;
};  // class blocks

void run_stage2(AppData& app_data,
                core::thread_pool& pool,
                const size_t n_threads) {
  const int first_index = 0;
  const int index_after_last = app_data.get_n_input();

  const my_blocks blks(first_index, index_after_last, n_threads);

  std::barrier bar(n_threads);

  kernels::dispatch_binning_pass(pool,
                                 n_threads,
                                 bar,
                                 app_data.get_n_input(),
                                 app_data.u_morton_keys.data(),
                                 app_data.u_morton_keys_alt.data(),
                                 0)
      .wait();
  kernels::dispatch_binning_pass(pool,
                                 n_threads,
                                 bar,
                                 app_data.get_n_input(),
                                 app_data.u_morton_keys_alt.data(),
                                 app_data.u_morton_keys.data(),
                                 8)
      .wait();
  kernels::dispatch_binning_pass(pool,
                                 n_threads,
                                 bar,
                                 app_data.get_n_input(),
                                 app_data.u_morton_keys.data(),
                                 app_data.u_morton_keys_alt.data(),
                                 16)
      .wait();
  kernels::dispatch_binning_pass(pool,
                                 n_threads,
                                 bar,
                                 app_data.get_n_input(),
                                 app_data.u_morton_keys_alt.data(),
                                 app_data.u_morton_keys.data(),
                                 24)
      .wait();
}

}  // namespace cpu
