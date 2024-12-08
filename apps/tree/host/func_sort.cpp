#include "func_sort.hpp"

#include <algorithm>
#include <barrier>
#include <condition_variable>
#include <numeric>

namespace cpu {

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

namespace kernels {

constexpr int BASE_BITS = 8;
constexpr int BASE = (1 << BASE_BITS);  // 256
constexpr int MASK = (BASE - 1);        // 0xFF

constexpr int DIGITS(const unsigned int v, const int shift) {
  return (v >> shift) & MASK;
}

// shared among threads
// need to reset 'bucket' and 'current_thread' before each pass
struct {
  std::mutex mtx;
  int bucket[BASE] = {};  // shared among threads
  std::condition_variable cv;
  size_t current_thread = 0;
} sort;

void k_binning_pass(const size_t tid,
                    std::barrier<>& barrier,
                    const uint32_t* u_sort_begin,
                    const uint32_t* u_sort_end,
                    uint32_t* u_sort_alt,  // output
                    const int shift) {
  // DEBUG_PRINT("[tid ", tid, "] started. (Binning, shift=", shift, ")");

  int local_bucket[BASE] = {};

  // compute histogram (local)
  std::for_each(
      u_sort_begin, u_sort_end, [shift, &local_bucket](const uint32_t& code) {
        ++local_bucket[DIGITS(code, shift)];
      });

  std::unique_lock lck(sort.mtx);

  // update to shared bucket
  for (auto i = 0; i < BASE; ++i) {
    sort.bucket[i] += local_bucket[i];
  }

  lck.unlock();

  barrier.arrive_and_wait();

  if (tid == 0) {
    std::partial_sum(std::begin(sort.bucket),
                     std::end(sort.bucket),
                     std::begin(sort.bucket));
  }

  barrier.arrive_and_wait();

  lck.lock();
  sort.cv.wait(lck, [&] { return tid == sort.current_thread; });

  // update the local_bucket from the shared bucket
  for (auto i = 0; i < BASE; i++) {
    sort.bucket[i] -= local_bucket[i];
    local_bucket[i] = sort.bucket[i];
  }

  --sort.current_thread;
  sort.cv.notify_all();

  lck.unlock();

  std::for_each(u_sort_begin, u_sort_end, [&](auto code) {
    u_sort_alt[local_bucket[DIGITS(code, shift)]++] = code;
  });

  // DEBUG_PRINT("[tid ", tid, "] ended. (Binning, shift=", shift, ")");
}

core::multi_future<void> dispatch_binning_pass(core::thread_pool& pool,
                                               const size_t n_threads,
                                               std::barrier<>& barrier,
                                               const int n,
                                               const uint32_t* u_sort,
                                               uint32_t* u_sort_alt,
                                               const int shift) {
  constexpr auto first_index = 0;
  const auto index_after_last = n;

  const my_blocks blks(first_index, index_after_last, n_threads);

  core::multi_future<void> future;
  future.futures.reserve(blks.get_num_blocks());

  std::fill_n(sort.bucket, BASE, 0);
  sort.current_thread = n_threads - 1;

  // I could have used the simpler API, but I need the 'blk' index for my kernel

  for (size_t blk = 0; blk < blks.get_num_blocks(); ++blk) {
    future.futures.push_back(pool.submit_task([start = blks.start(blk),
                                               end = blks.end(blk),
                                               blk,
                                               &barrier,
                                               u_sort,
                                               u_sort_alt,
                                               shift] {
      k_binning_pass(static_cast<int>(blk),
                     barrier,
                     u_sort + start,
                     u_sort + end,
                     u_sort_alt,
                     shift);
    }));
  }

  return future;
}

}  // namespace kernels

}  // namespace cpu
