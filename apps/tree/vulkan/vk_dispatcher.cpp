#include "vk_dispatcher.hpp"

namespace vulkan {

Dispatcher::Dispatcher(Engine &engine, AppData &app_data)
    : engine_ref(engine),
      app_data_ref(app_data),
      tmp_storage(engine.get_mr(), app_data.get_n_input()) {
  // --------------------------------------------------------------------------
  // Morton
  // --------------------------------------------------------------------------

  auto morton_algo =
      engine
          .algorithm("tree_morton.comp",
                     {
                         engine.get_buffer(app_data.u_input_points.data()),
                         engine.get_buffer(app_data.u_morton_keys.data()),
                     })
          ->set_push_constants<MortonPushConstants>({
              .n = static_cast<uint32_t>(app_data.get_n_input()),
              .min_coord = app_data.min_coord,
              .range = app_data.range,
          })
          ->build();

  cached_algorithms.try_emplace("morton", std::move(morton_algo));

  // --------------------------------------------------------------------------
  // Merge Sort
  // --------------------------------------------------------------------------

  auto merge_sort_algo =
      engine
          .algorithm("tree_merge_sort.comp",
                     {
                         engine.get_buffer(app_data.u_morton_keys.data()),
                         engine.get_buffer(app_data.u_morton_keys_alt.data()),
                     })
          ->set_push_constants<MergeSortPushConstants>({
              .n_logical_blocks = 16,
              .n = static_cast<uint32_t>(app_data.get_n_input()),
              .width = 16,
              .num_pairs = 8,
          })
          ->build();

  cached_algorithms.try_emplace("merge_sort", std::move(merge_sort_algo));

  // --------------------------------------------------------------------------
  // Find Dups
  // --------------------------------------------------------------------------

  auto find_dups_algo =
      engine
          .algorithm("tree_find_dups.comp",
                     {
                         engine.get_buffer(app_data.u_morton_keys_alt.data()),
                         engine.get_buffer(tmp_storage.u_contributes.data()),
                     })
          ->set_push_constants<FindDupsPushConstants>({
              .n = static_cast<int32_t>(app_data.get_n_input()),
          })
          ->build();

  cached_algorithms.try_emplace("find_dups", std::move(find_dups_algo));

  // --------------------------------------------------------------------------
  // Move Dups
  // --------------------------------------------------------------------------

  auto move_dups_algo =
      engine
          .algorithm("tree_move_dups.comp",
                     {
                         engine.get_buffer(tmp_storage.u_contributes.data()),
                         engine.get_buffer(app_data.u_morton_keys_alt.data()),
                         engine.get_buffer(app_data.u_morton_keys.data()),
                     })
          ->set_push_constants<MoveDupsPushConstants>({
              .n = static_cast<uint32_t>(app_data.get_n_input()),
          })
          ->build();

  cached_algorithms.try_emplace("move_dups", std::move(move_dups_algo));

  // --------------------------------------------------------------------------
  // Build Radix Tree
  // --------------------------------------------------------------------------

  auto build_radix_tree_algo =
      engine
          .algorithm(
              "tree_build_radix_tree.comp",
              {
                  engine.get_buffer(app_data.u_morton_keys.data()),
                  engine.get_buffer(app_data.brt.u_prefix_n.data()),
                  engine.get_buffer(app_data.brt.u_has_leaf_left.data()),
                  engine.get_buffer(app_data.brt.u_has_leaf_right.data()),
                  engine.get_buffer(app_data.brt.u_left_child.data()),
                  engine.get_buffer(app_data.brt.u_parents.data()),
              })
          ->place_holder_push_constants<BuildTreePushConstants>()
          ->build();

  cached_algorithms.try_emplace("build_radix_tree",
                                std::move(build_radix_tree_algo));

  // --------------------------------------------------------------------------
  // Edge Count
  // --------------------------------------------------------------------------

  auto edge_count_algo =
      engine
          .algorithm("tree_edge_count.comp",
                     {
                         engine.get_buffer(app_data.brt.u_prefix_n.data()),
                         engine.get_buffer(app_data.brt.u_parents.data()),
                         engine.get_buffer(app_data.u_edge_count.data()),
                     })
          ->place_holder_push_constants<EdgeCountPushConstants>()
          ->build();

  cached_algorithms.try_emplace("edge_count", std::move(edge_count_algo));

  // --------------------------------------------------------------------------
  // Prefix Sum
  // --------------------------------------------------------------------------

  auto prefix_sum_algo =
      engine
          .algorithm("tree_naive_prefix_sum.comp",
                     {
                         engine.get_buffer(app_data.u_edge_count.data()),
                         engine.get_buffer(app_data.u_edge_offset.data()),
                     })
          ->place_holder_push_constants<PrefixSumPushConstants>()
          ->build();

  cached_algorithms.try_emplace("prefix_sum", std::move(prefix_sum_algo));

  // --------------------------------------------------------------------------
  // Build Octree
  // --------------------------------------------------------------------------

  auto build_octree_algo =
      engine
          .algorithm(
              "tree_build_octree.comp",
              {
                  engine.get_buffer(app_data.oct.u_children.data()),
                  engine.get_buffer(app_data.oct.u_corner.data()),
                  engine.get_buffer(app_data.oct.u_cell_size.data()),
                  engine.get_buffer(app_data.oct.u_child_node_mask.data()),
                  engine.get_buffer(app_data.oct.u_child_leaf_mask.data()),
                  engine.get_buffer(app_data.u_edge_offset.data()),
                  engine.get_buffer(app_data.u_edge_count.data()),
                  engine.get_buffer(app_data.u_morton_keys.data()),
                  engine.get_buffer(app_data.brt.u_prefix_n.data()),
                  engine.get_buffer(app_data.brt.u_parents.data()),
                  engine.get_buffer(app_data.brt.u_left_child.data()),
                  engine.get_buffer(app_data.brt.u_has_leaf_left.data()),
                  engine.get_buffer(app_data.brt.u_has_leaf_right.data()),
              })
          ->place_holder_push_constants<OctreePushConstants>()
          ->build();

  cached_algorithms.try_emplace("build_octree", std::move(build_octree_algo));
}

// ----------------------------------------------------------------------------
// Stage 1: Morton
// ----------------------------------------------------------------------------

void Dispatcher::run_stage1(Sequence *seq) {
  const int total_iterations = app_data_ref.get_n_input();

  auto algo = cached_algorithms.at("morton").get();

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 2: Merge Sort
// ----------------------------------------------------------------------------

void Dispatcher::run_stage2(Sequence *seq) {
  const uint32_t n = app_data_ref.get_n_input();

  auto algo = cached_algorithms.at("merge_sort").get();

  constexpr auto threads_per_block = 256;

  // Make copies of the input and output buffers
  auto iterations = 0;
  for (uint32_t width = 1; width < n; width *= 2, ++iterations) {
    uint32_t num_pairs = (n + 2 * width - 1) / (2 * width);
    uint32_t total_threads = num_pairs;
    uint32_t logical_blocks =
        (total_threads + threads_per_block - 1) / threads_per_block;

    algo->set_push_constants(MergeSortPushConstants{
        .n_logical_blocks = logical_blocks,
        .n = n,
        .width = width,
        .num_pairs = num_pairs,
    });

    if (iterations % 2 == 0) {
      algo->update_descriptor_sets({
          engine_ref.get_buffer(app_data_ref.u_morton_keys.data()),
          engine_ref.get_buffer(app_data_ref.u_morton_keys_alt.data()),
      });
    } else {
      algo->update_descriptor_sets({
          engine_ref.get_buffer(app_data_ref.u_morton_keys_alt.data()),
          engine_ref.get_buffer(app_data_ref.u_morton_keys.data()),
      });
    }

    seq->record_commands(algo, n);
    seq->launch_kernel_async();
    seq->sync();

    // std::swap(elements_in_copy, elements_out_copy);
  }

  // If the number of iterations is odd, swap the input and output buffers
  if (iterations % 2 == 0) {
    std::swap(app_data_ref.u_morton_keys, app_data_ref.u_morton_keys_alt);
  }

  const auto is_sorted =
      std::is_sorted(app_data_ref.get_sorted_morton_keys(),
                     app_data_ref.get_sorted_morton_keys() + n);
  spdlog::info("Is sorted: {}", is_sorted);

  exit(0);
}

}  // namespace vulkan
