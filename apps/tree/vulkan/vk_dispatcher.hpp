#pragma once

#include "../app_data.hpp"
#include "redwood/vulkan/engine.hpp"
#include "tmp_storage.hpp"

namespace vulkan {

struct Dispatcher {
  explicit Dispatcher(Engine &engine, AppData &app_data);

  // --------------------------------------------------------------------------
  // Stage 1
  // --------------------------------------------------------------------------

  struct MortonPushConstants {
    uint32_t n;
    float min_coord;
    float range;
  };

  // --------------------------------------------------------------------------
  // Stage 2
  // --------------------------------------------------------------------------

  struct MergeSortPushConstants {
    uint32_t n_logical_blocks;
    uint32_t n;
    uint32_t width;
    uint32_t num_pairs;
  };

  // --------------------------------------------------------------------------
  // Stage 3
  // --------------------------------------------------------------------------

  struct FindDupsPushConstants {
    int32_t n;
  };

  struct MoveDupsPushConstants {
    uint32_t n;
  };

  // --------------------------------------------------------------------------
  // Stage 4
  // --------------------------------------------------------------------------

  struct BuildTreePushConstants {
    int32_t n;
  };

  // --------------------------------------------------------------------------
  // Stage 5
  // --------------------------------------------------------------------------

  struct EdgeCountPushConstants {
    int32_t n_brt_nodes;
  };

  // --------------------------------------------------------------------------
  // Stage 6
  // --------------------------------------------------------------------------

  struct PrefixSumPushConstants {
    uint32_t inputSize;
  };

  // --------------------------------------------------------------------------
  // Stage 7
  // --------------------------------------------------------------------------

  struct OctreePushConstants {
    float min_coord;
    float range;
    int32_t n_brt_nodes;
  };

  void run_stage1(Sequence *seq);
  void run_stage2(Sequence *seq);
  void run_stage3(Sequence *seq);
  void run_stage4(Sequence *seq);
  void run_stage5(Sequence *seq);
  void run_stage6(Sequence *seq);
  void run_stage7(Sequence *seq);

  Engine &engine_ref;
  AppData &app_data_ref;
  TmpStorage tmp_storage;

  std::unordered_map<std::string, std::shared_ptr<Algorithm>> algorithms;
};

}  // namespace vulkan
