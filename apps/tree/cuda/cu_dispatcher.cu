#include <spdlog/spdlog.h>

#include <cub/util_math.cuh>
#include <numeric>

#include "01_morton.cuh"
#include "02_sort.cuh"
#include "03_unique.cuh"
#include "04_radix_tree.cuh"
#include "05_edge_count.cuh"
#include "06_prefix_sum.cuh"
#include "07_octree.cuh"
#include "agents/prefix_sum_agent.cuh"
#include "agents/unique_agent.cuh"
#include "common.cuh"
#include "cu_dispatcher.cuh"
#include "im_storage.cuh"
#include "redwood/cuda/helpers.cuh"

namespace cuda {

namespace kernels {

// Number of digit bins
constexpr auto RADIX = 256;
constexpr auto RADIX_MASK = RADIX - 1;
constexpr auto RADIX_LOG = 8;

constexpr auto SEC_RADIX_START = 1 * RADIX;     // 256
constexpr auto THIRD_RADIX_START = 2 * RADIX;   // 512
constexpr auto FOURTH_RADIX_START = 3 * RADIX;  // 768

// For the upfront global histogram kernel
// #define G_HIST_PART_SIZE 65536
// #define G_HIST_VEC_SIZE 16384

constexpr auto G_HIST_PART_SIZE = 65536;
constexpr auto G_HIST_VEC_SIZE = 16384;

// For the digit binning
// Partition tile size in k_DigitBinning

// #define BIN_PART_SIZE 7680

// looks like we use 512 threads per block, and 15 items per thread, so 7680
constexpr auto BIN_WARPS = 512 / LANE_COUNT;  // 16;
constexpr auto BIN_KEYS_PER_THREAD = 15;
constexpr auto BIN_SUB_PART_SIZE = BIN_KEYS_PER_THREAD * LANE_COUNT;  // 480;

constexpr auto BIN_PART_SIZE = 512 * BIN_KEYS_PER_THREAD;  // 7680;

// Total size of warp histograms in shared memory in k_DigitBinning
// #define BIN_HISTS_SIZE 4096
constexpr auto BIN_HISTS_SIZE = 4096;

// Subpartition tile size of a single warp in k_DigitBinning
// #define BIN_SUB_PART_SIZE 480

// Warps per threadblock in k_DigitBinning
// Keys per thread in k_DigitBinning

// Starting offset of a subpartition tile
#define BIN_SUB_PART_START (WARP_INDEX * BIN_SUB_PART_SIZE)

// Starting offset of a partition tile
#define BIN_PART_START (partitionIndex * BIN_PART_SIZE)

// Flag value inidicating neither inclusive sum, nor reduction of a
// partition tile is ready
// for the chained scan with decoupled lookback
#define FLAG_NOT_READY 0
// Flag value indicating reduction of a partition tile is ready
#define FLAG_REDUCTION 1
// Flag value indicating inclusive sum of a partition tile is ready
#define FLAG_INCLUSIVE 2
#define FLAG_MASK 3

// ============================================================================
// Kernel entry points
// ============================================================================

__global__ void k_GlobalHistogram(const unsigned int *sort,
                                  unsigned int *global_histogram,
                                  unsigned int size) {
  __shared__ unsigned int s_globalHistFirst[RADIX * 2];
  __shared__ unsigned int s_globalHistSec[RADIX * 2];
  __shared__ unsigned int s_globalHistThird[RADIX * 2];
  __shared__ unsigned int s_globalHistFourth[RADIX * 2];

  const auto logicalBlocks = cub::DivideAndRoundUp(size, G_HIST_PART_SIZE);

  for (auto yanwen_block_id = blockIdx.x; yanwen_block_id < logicalBlocks;
       yanwen_block_id += gridDim.x) {
    // clear shared memory
    for (unsigned int i = threadIdx.x; i < RADIX * 2; i += blockDim.x) {
      s_globalHistFirst[i] = 0;
      s_globalHistSec[i] = 0;
      s_globalHistThird[i] = 0;
      s_globalHistFourth[i] = 0;
    }
    __syncthreads();

    // histogram
    {
      // 64 threads : 1 histogram in shared memory

      // clang-format off
       unsigned int* s_wavesHistFirst = &s_globalHistFirst[threadIdx.x / 64 * RADIX];
       unsigned int* s_wavesHistSec = &s_globalHistSec[threadIdx.x / 64 * RADIX];
       unsigned int* s_wavesHistThird = &s_globalHistThird[threadIdx.x / 64 * RADIX];
       unsigned int* s_wavesHistFourth = &s_globalHistFourth[threadIdx.x / 64 * RADIX];
      // clang-format on

      if (yanwen_block_id < logicalBlocks - 1) {
        const unsigned int partEnd = (yanwen_block_id + 1) * G_HIST_VEC_SIZE;
        for (unsigned int i = threadIdx.x + (yanwen_block_id * G_HIST_VEC_SIZE);
             i < partEnd;
             i += blockDim.x) {
          uint4 t[1] = {reinterpret_cast<const uint4 *>(sort)[i]};

          atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t *>(t)[0]], 1);
          atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t *>(t)[1]], 1);
          atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t *>(t)[2]], 1);
          atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t *>(t)[3]], 1);

          atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t *>(t)[4]], 1);
          atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t *>(t)[5]], 1);
          atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t *>(t)[6]], 1);
          atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t *>(t)[7]], 1);

          atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t *>(t)[8]], 1);
          atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t *>(t)[9]], 1);
          atomicAdd(&s_globalHistThird[reinterpret_cast<uint8_t *>(t)[10]], 1);
          atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t *>(t)[11]], 1);

          atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t *>(t)[12]], 1);
          atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t *>(t)[13]], 1);
          atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t *>(t)[14]], 1);
          atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t *>(t)[15]], 1);
        }
      }

      if (yanwen_block_id == logicalBlocks - 1) {
        for (unsigned int i =
                 threadIdx.x + (yanwen_block_id * G_HIST_PART_SIZE);
             i < size;
             i += blockDim.x) {
          unsigned int t[1] = {sort[i]};
          atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t *>(t)[0]], 1);
          atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t *>(t)[1]], 1);
          atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t *>(t)[2]], 1);
          atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t *>(t)[3]], 1);
        }
      }
    }
    __syncthreads();

    // reduce and add to device
    for (unsigned int i = threadIdx.x; i < RADIX; i += blockDim.x) {
      atomicAdd(&global_histogram[i],
                s_globalHistFirst[i] + s_globalHistFirst[i + RADIX]);
      atomicAdd(&global_histogram[i + SEC_RADIX_START],
                s_globalHistSec[i] + s_globalHistSec[i + RADIX]);
      atomicAdd(&global_histogram[i + THIRD_RADIX_START],
                s_globalHistThird[i] + s_globalHistThird[i + RADIX]);
      atomicAdd(&global_histogram[i + FOURTH_RADIX_START],
                s_globalHistFourth[i] + s_globalHistFourth[i + RADIX]);
    }
  }
}

// fixed to use 4 blocks, and 'radix' (256) threads
__global__ void k_Scan(const unsigned int *globalHistogram,
                       unsigned int *firstPassHistogram,
                       unsigned int *secPassHistogram,
                       unsigned int *thirdPassHistogram,
                       unsigned int *fourthPassHistogram) {
  __shared__ unsigned int s_scan[RADIX];

  s_scan[threadIdx.x] = InclusiveWarpScanCircularShift(
      globalHistogram[threadIdx.x + blockIdx.x * RADIX]);
  __syncthreads();

  if (threadIdx.x < (RADIX >> LANE_LOG))
    s_scan[threadIdx.x << LANE_LOG] =
        ActiveExclusiveWarpScan(s_scan[threadIdx.x << LANE_LOG]);
  __syncthreads();

  switch (blockIdx.x) {
    case 0:
      firstPassHistogram[threadIdx.x] =
          (s_scan[threadIdx.x] +
           (getLaneId() ? __shfl_sync(0xfffffffe, s_scan[threadIdx.x - 1], 1)
                        : 0))
              << 2 |
          FLAG_INCLUSIVE;
      break;
    case 1:
      secPassHistogram[threadIdx.x] =
          (s_scan[threadIdx.x] +
           (getLaneId() ? __shfl_sync(0xfffffffe, s_scan[threadIdx.x - 1], 1)
                        : 0))
              << 2 |
          FLAG_INCLUSIVE;
      break;
    case 2:
      thirdPassHistogram[threadIdx.x] =
          (s_scan[threadIdx.x] +
           (getLaneId() ? __shfl_sync(0xfffffffe, s_scan[threadIdx.x - 1], 1)
                        : 0))
              << 2 |
          FLAG_INCLUSIVE;
      break;
    case 3:
      fourthPassHistogram[threadIdx.x] =
          (s_scan[threadIdx.x] +
           (getLaneId() ? __shfl_sync(0xfffffffe, s_scan[threadIdx.x - 1], 1)
                        : 0))
              << 2 |
          FLAG_INCLUSIVE;
      break;
    default:
      break;
  }
}

// ============================================================================
// Yanwen's version
//
// ============================================================================

__global__ void k_DigitBinningPass(unsigned int *sort,
                                   unsigned int *alt,
                                   volatile unsigned int *passHistogram,
                                   volatile unsigned int *index,
                                   unsigned int size,
                                   unsigned int radixShift) {
  __shared__ unsigned int s_warpHistograms[BIN_PART_SIZE];
  __shared__ unsigned int s_localHistogram[RADIX];

  volatile unsigned int *s_warpHist =
      &s_warpHistograms[WARP_INDEX << RADIX_LOG];

  const auto logicalBlocks = cub::DivideAndRoundUp(size, BIN_PART_SIZE);

  for (auto yanwen_block_id = blockIdx.x; yanwen_block_id < logicalBlocks;
       yanwen_block_id += gridDim.x) {
    // clear shared memory
    for (unsigned int i = threadIdx.x; i < BIN_HISTS_SIZE;
         i += blockDim.x)  // unnecessary work for last partion but still a win
                           // to avoid another barrier
      s_warpHistograms[i] = 0;

    // atomically assign partition tiles
    if (threadIdx.x == 0)
      s_warpHistograms[BIN_PART_SIZE - 1] =
          atomicAdd((unsigned int *)&index[radixShift >> 3], 1);
    __syncthreads();
    const unsigned int partitionIndex = s_warpHistograms[BIN_PART_SIZE - 1];

    // To handle input sizes not perfect multiples of the partition tile size
    if (partitionIndex < logicalBlocks - 1) {
      // load keys
      unsigned int keys[BIN_KEYS_PER_THREAD];
#pragma unroll
      for (unsigned int i = 0,
                        t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START;
           i < BIN_KEYS_PER_THREAD;
           ++i, t += LANE_COUNT)
        keys[i] = sort[t];

      uint16_t offsets[BIN_KEYS_PER_THREAD];

// WLMS
#pragma unroll
      for (unsigned int i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
        // CUB version "match any"
        /*
        unsigned warpFlags;
        #pragma unroll
        for (int k = 0; k < RADIX_LOG; ++k)
        {
            unsigned int mask;
            unsigned int current_bit = 1 << k + radixShift;
            asm("{\n"
                "    .reg .pred p;\n"
                "    and.b32 %0, %1, %2;"
                "    setp.ne.u32 p, %0, 0;\n"
                "    vote.ballot.sync.b32 %0, p, 0xffffffff;\n"
                "    @!p not.b32 %0, %0;\n"
                "}\n" : "=r"(mask) : "r"(keys[i]), "r"(current_bit));
            warpFlags = (k == 0) ? mask : warpFlags & mask;
        }
        const unsigned int bits = __popc(warpFlags & getLaneMaskLt());
        */
        unsigned warpFlags = 0xffffffff;
#pragma unroll
        for (int k = 0; k < RADIX_LOG; ++k) {
          const bool t2 = keys[i] >> k + radixShift & 1;
          warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
        }
        const unsigned int bits = __popc(warpFlags & getLaneMaskLt());

        // An alternative, but slightly slower version.
        /*
        offsets[i] = s_warpHist[keys[i] >> radixShift & RADIX_MASK] + bits;
        __syncwarp(0xffffffff);
        if (bits == 0)
            s_warpHist[keys[i] >> radixShift & RADIX_MASK] +=
            __popc(warpFlags);
        __syncwarp(0xffffffff);
        */
        unsigned int preIncrementVal;
        if (bits == 0)
          preIncrementVal = atomicAdd(
              (unsigned int *)&s_warpHist[keys[i] >> radixShift & RADIX_MASK],
              __popc(warpFlags));

        offsets[i] =
            __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) +
            bits;
      }
      __syncthreads();

      // exclusive prefix sum up the warp histograms
      if (threadIdx.x < RADIX) {
        unsigned int reduction = s_warpHistograms[threadIdx.x];
        for (unsigned int i = threadIdx.x + RADIX; i < BIN_HISTS_SIZE;
             i += RADIX) {
          reduction += s_warpHistograms[i];
          s_warpHistograms[i] = reduction - s_warpHistograms[i];
        }

        atomicAdd((unsigned int *)&passHistogram[threadIdx.x +
                                                 (partitionIndex + 1) * RADIX],
                  FLAG_REDUCTION | reduction << 2);

        // begin the exclusive prefix sum across the reductions
        s_localHistogram[threadIdx.x] =
            InclusiveWarpScanCircularShift(reduction);
      }
      __syncthreads();

      if (threadIdx.x < (RADIX >> LANE_LOG))
        s_localHistogram[threadIdx.x << LANE_LOG] =
            ActiveExclusiveWarpScan(s_localHistogram[threadIdx.x << LANE_LOG]);
      __syncthreads();

      if (threadIdx.x < RADIX && getLaneId())
        s_localHistogram[threadIdx.x] +=
            __shfl_sync(0xfffffffe, s_localHistogram[threadIdx.x - 1], 1);
      __syncthreads();

      // update offsets
      if (WARP_INDEX) {
#pragma unroll
        for (unsigned int i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
          const unsigned int t2 = keys[i] >> radixShift & RADIX_MASK;
          offsets[i] += s_warpHist[t2] + s_localHistogram[t2];
        }
      } else {
#pragma unroll
        for (unsigned int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
          offsets[i] += s_localHistogram[keys[i] >> radixShift & RADIX_MASK];
      }
      __syncthreads();

// scatter keys into shared memory
#pragma unroll
      for (unsigned int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
        s_warpHistograms[offsets[i]] = keys[i];

      // split the warps into single thread cooperative groups and lookback
      if (threadIdx.x < RADIX) {
        unsigned int reduction = 0;
        for (unsigned int k = partitionIndex; k >= 0;) {
          const unsigned int flagPayload =
              passHistogram[threadIdx.x + k * RADIX];

          if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
            reduction += flagPayload >> 2;
            atomicAdd(
                (unsigned int *)&passHistogram[threadIdx.x +
                                               (partitionIndex + 1) * RADIX],
                1 | (reduction << 2));
            s_localHistogram[threadIdx.x] =
                reduction - s_localHistogram[threadIdx.x];
            break;
          }

          if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION) {
            reduction += flagPayload >> 2;
            k--;
          }
        }
      }
      __syncthreads();

// scatter runs of keys into device memory
#pragma unroll
      for (unsigned int i = threadIdx.x; i < BIN_PART_SIZE; i += blockDim.x)
        alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] +
            i] = s_warpHistograms[i];
    }

    // Process the final partition slightly differently
    if (partitionIndex == logicalBlocks - 1) {
      // immediately begin lookback
      if (threadIdx.x < RADIX) {
        if (partitionIndex) {
          unsigned int reduction = 0;
          for (unsigned int k = partitionIndex; k >= 0;) {
            const unsigned int flagPayload =
                passHistogram[threadIdx.x + k * RADIX];

            if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
              reduction += flagPayload >> 2;
              s_localHistogram[threadIdx.x] = reduction;
              break;
            }

            if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION) {
              reduction += flagPayload >> 2;
              k--;
            }
          }
        } else {
          s_localHistogram[threadIdx.x] = passHistogram[threadIdx.x] >> 2;
        }
      }
      __syncthreads();

      const unsigned int partEnd = BIN_PART_START + BIN_PART_SIZE;
      for (unsigned int i = threadIdx.x + BIN_PART_START; i < partEnd;
           i += blockDim.x) {
        unsigned int key;
        unsigned int offset;
        unsigned warpFlags = 0xffffffff;

        if (i < size) key = sort[i];

#pragma unroll
        for (unsigned int k = 0; k < RADIX_LOG; ++k) {
          const bool t = key >> k + radixShift & 1;
          warpFlags &= (t ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t);
        }
        const unsigned int bits = __popc(warpFlags & getLaneMaskLt());

#pragma unroll
        for (unsigned int k = 0; k < BIN_WARPS; ++k) {
          unsigned int preIncrementVal;
          if (WARP_INDEX == k && bits == 0 && i < size)
            preIncrementVal =
                atomicAdd(&s_localHistogram[key >> radixShift & RADIX_MASK],
                          __popc(warpFlags));

          if (WARP_INDEX == k)
            offset =
                __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) +
                bits;
          __syncthreads();
        }

        if (i < size) alt[offset] = key;
      }
    }
  }
}

// //
// ----------------------------------------------------------------------------
// // Unique
// //
// ----------------------------------------------------------------------------

// __global__ void k_FindDups(const unsigned int *u_keys,
//                            int *u_flag_heads,
//                            const int n) {
//   __shared__ agents::UniqueAgent::TempStorage temp_storage;

//   agents::UniqueAgent agent(n);
//   agent.Process_FindDups(temp_storage, u_keys, u_flag_heads, n);
// }

// __global__ void k_MoveDups(const unsigned int *u_keys,
//                            const int *u_flag_heads_sums,
//                            const int n,
//                            unsigned int *u_keys_out,
//                            int *n_unique_out) {
//   agents::UniqueAgent agent(n);
//   agent.Process_MoveDups(
//       u_keys, u_flag_heads_sums, n, u_keys_out, n_unique_out);
// }

}  // namespace kernels

// ----------------------------------------------------------------------------
// Stage 1 (input -> morton code)
// ----------------------------------------------------------------------------

void run_stage1(AppData &app_data, cudaStream_t stream) {
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

void run_stage2(AppData &app_data, ImStorage &im_storage, cudaStream_t stream) {
  const auto n = app_data.get_n_input();

  im_storage.clearSmem();

  const auto smem = 0;
  constexpr auto grid_size = 16;

  kernels::k_GlobalHistogram<<<grid_size,
                               ImStorage::GLOBAL_HIST_THREADS,
                               smem,
                               stream>>>(app_data.u_morton_keys.data(),
                                         im_storage.d_global_histogram,
                                         (int)app_data.get_n_input());

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

// ----------------------------------------------------------------------------
// Stage 3 (unique) (sorted morton code -> unique sorted morton code)
// ----------------------------------------------------------------------------

void run_stage3(AppData &app_data, ImStorage &im_storage, cudaStream_t stream) {
  constexpr auto unique_block_size = agents::UniqueAgent::n_threads;  // 256
  constexpr auto prefix_block_size =
      agents::PrefixSumAgent<unsigned int>::n_threads;  // 128

  constexpr auto grid_size = 16;

  kernels::k_FindDups<<<grid_size, unique_block_size, 0, stream>>>(
      app_data.get_sorted_morton_keys(),
      im_storage.u_flag_heads,  // <-- output
      app_data.get_n_input());

  CUDA_CHECK(cudaStreamSynchronize(stream));

  ::std::partial_sum(im_storage.u_flag_heads,
                     im_storage.u_flag_heads + app_data.get_n_input(),
                     im_storage.u_flag_heads);

  kernels::k_MoveDups<<<grid_size, unique_block_size, 0, stream>>>(
      app_data.get_sorted_morton_keys(),
      im_storage.u_flag_heads,
      app_data.get_n_input(),
      app_data.get_unique_morton_keys(),  // <-- output
      nullptr);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  const auto n_unique = im_storage.u_flag_heads[app_data.get_n_input() - 1] + 1;
  app_data.set_n_unique(n_unique);
  app_data.set_n_brt_nodes(n_unique - 1);
}

// ----------------------------------------------------------------------------
// Stage 4 (build tree) (unique sorted morton code -> tree nodes)
// ----------------------------------------------------------------------------

void run_stage4(AppData &app_data, cudaStream_t stream) {
  constexpr auto gridDim = 16;
  constexpr auto blockDim = 512;
  constexpr auto sharedMem = 0;

  kernels::k_BuildRadixTree<<<gridDim, blockDim, sharedMem, stream>>>(
      app_data.get_n_unique(),
      app_data.get_unique_morton_keys(),
      app_data.brt.u_prefix_n.data(),
      app_data.brt.u_has_leaf_left.data(),
      app_data.brt.u_has_leaf_right.data(),
      app_data.brt.u_left_child.data(),
      app_data.brt.u_parents.data());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ----------------------------------------------------------------------------
// Stage 5 (edge count) (tree nodes -> edge count)
// ----------------------------------------------------------------------------

void run_stage5(AppData &app_data, cudaStream_t stream) {
  constexpr auto gridDim = 16;
  constexpr auto blockDim = 512;
  constexpr auto sharedMem = 0;

  kernels::k_EdgeCount<<<gridDim, blockDim, sharedMem, stream>>>(
      app_data.brt.u_prefix_n.data(),
      app_data.brt.u_parents.data(),
      app_data.u_edge_count.data(),
      app_data.get_n_brt_nodes());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ----------------------------------------------------------------------------
// Stage 6 (edge offset) (edge count -> edge offset)
// ----------------------------------------------------------------------------

void run_stage6(AppData &app_data, cudaStream_t stream) {
  constexpr auto n_threads = agents::PrefixSumAgent<int>::n_threads;

  // has to be single block
  kernels::k_SingleBlockExclusiveScan<<<1, n_threads, 0, stream>>>(
      app_data.u_edge_count.data(),
      app_data.u_edge_offset.data(),  // <-- output
      app_data.get_n_brt_nodes());

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // num oct is the result of last of prefix sum
  app_data.set_n_octree_nodes(
      app_data.u_edge_offset[app_data.get_n_brt_nodes() - 1]);
}

// ----------------------------------------------------------------------------
// Stage 7 (octree) (everything above -> octree)
// ----------------------------------------------------------------------------

void run_stage7(AppData &app_data, cudaStream_t stream) {
  constexpr auto block_size = 512;
  //  k_MakeOctNodes<<<grid_size, block_size, 0, stream>>>(
  //       pipe->oct.u_children,
  //       pipe->oct.u_corner,
  //       pipe->oct.u_cell_size,
  //       pipe->oct.u_child_node_mask,
  //       pipe->u_edge_offsets,
  //       pipe->u_edge_counts,
  //       pipe->getUniqueKeys(),
  //       pipe->brt.u_prefix_n,
  //       pipe->brt.u_parents,
  //       pipe->min_coord,
  //       pipe->range,
  //       pipe->n_brt_nodes());

  //   k_LinkLeafNodes<<<grid_size, block_size, 0, stream>>>(
  //       pipe->oct.u_children,
  //       pipe->oct.u_child_node_mask,
  //       pipe->u_edge_offsets,
  //       pipe->u_edge_counts,
  //       pipe->getUniqueKeys(),
  //       pipe->brt.u_has_leaf_left,
  //       pipe->brt.u_has_leaf_right,
  //       pipe->brt.u_prefix_n,
  //       pipe->brt.u_parents,
  //       pipe->brt.u_left_child,
  //       pipe->n_brt_nodes());
  constexpr auto gridDim = 16;
  constexpr auto blockDim = 512;
  constexpr auto sharedMem = 0;

  kernels::k_MakeOctNodes<<<gridDim, blockDim, sharedMem, stream>>>(
      reinterpret_cast<int(*)[8]>(app_data.oct.u_children.data()),
      app_data.oct.u_corner.data(),
      app_data.oct.u_cell_size.data(),
      app_data.oct.u_child_node_mask.data(),
      app_data.u_edge_offset.data(),
      app_data.u_edge_count.data(),
      app_data.get_unique_morton_keys(),
      app_data.brt.u_prefix_n.data(),
      app_data.brt.u_parents.data(),
      app_data.min_coord,
      app_data.range,
      app_data.get_n_brt_nodes());

  //   kernels::k_LinkLeafNodes<<<gridDim, blockDim, sharedMem, stream>>>(
  //       reinterpret_cast<int(*)[8]>(app_data.oct.u_children.data()),
  //       app_data.oct.u_child_node_mask.data(),
  //       app_data.u_edge_offset.data(),
  //       app_data.u_edge_count.data(),
  //       app_data.get_unique_morton_keys(),
  //       app_data.brt.u_has_leaf_left.data(),
  //       app_data.brt.u_has_leaf_right.data(),
  //       app_data.brt.u_prefix_n.data(),
  //       app_data.brt.u_parents.data(),
  //       app_data.brt.u_left_child.data(),
  //       app_data.get_n_brt_nodes());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace cuda
