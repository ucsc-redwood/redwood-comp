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
                         engine.get_buffer(app_data.get_sorted_morton_keys()),
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
                         engine.get_buffer(tmp_storage.u_out_idx.data()),
                         engine.get_buffer(app_data.get_sorted_morton_keys()),
                         engine.get_buffer(app_data.get_unique_morton_keys()),
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
                  engine.get_buffer(app_data.get_unique_morton_keys()),
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
                  engine.get_buffer(app_data.get_unique_morton_keys()),
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

  //   exit(0);
}

// ----------------------------------------------------------------------------
// Stage 3: Unique
// ----------------------------------------------------------------------------

void Dispatcher::run_stage3(Sequence *seq) {
  const uint32_t n = app_data_ref.get_n_input();

  auto find_dups = cached_algorithms.at("find_dups").get();

  auto prefix_sum = cached_algorithms.at("prefix_sum").get();

  auto move_dups = cached_algorithms.at("move_dups").get();

  find_dups->update_push_constants(FindDupsPushConstants{
      .n = static_cast<int32_t>(n),
  });

  prefix_sum->update_push_constants(PrefixSumPushConstants{
      .inputSize = n,
  });
  prefix_sum->update_descriptor_sets({
      engine_ref.get_buffer(tmp_storage.u_contributes.data()),
      engine_ref.get_buffer(tmp_storage.u_out_idx.data()),
  });

  move_dups->update_push_constants(MoveDupsPushConstants{
      .n = n,
  });

  seq->record_commands(find_dups, n);
  seq->launch_kernel_async();
  seq->sync();

  // print 10 u_contributes
  for (auto i = 0; i < 10; ++i) {
    spdlog::trace("u_contributes[{}] = {}", i, tmp_storage.u_contributes[i]);
  }

  seq->record_commands_with_blocks(prefix_sum, 1);
  seq->launch_kernel_async();
  seq->sync();

  // print 10 u_out_idx
  for (auto i = 0; i < 10; ++i) {
    spdlog::trace("u_out_idx[{}] = {}", i, tmp_storage.u_out_idx[i]);
  }

  seq->record_commands(move_dups, n);
  seq->launch_kernel_async();
  seq->sync();

  const auto n_unique = tmp_storage.u_out_idx[n - 1] + 1;
  app_data_ref.set_n_unique(n_unique);
  app_data_ref.set_n_brt_nodes(n_unique - 1);

  // print 10 u_out_idx
  for (auto i = 0; i < 10; ++i) {
    spdlog::trace("u_out_idx[{}] = {}", i, tmp_storage.u_out_idx[i]);
  }

  spdlog::info("GPU n_unique: {}", n_unique);
}

// ----------------------------------------------------------------------------
// Stage 4: Build Radix Tree
// ----------------------------------------------------------------------------

void Dispatcher::run_stage4(Sequence *seq) {
  const int32_t n = app_data_ref.get_n_unique();

  auto build_radix_tree = cached_algorithms.at("build_radix_tree").get();

  build_radix_tree->update_push_constants(BuildTreePushConstants{
      .n = n,
  });

  seq->record_commands(build_radix_tree, n);
  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 5: Edge Count
// ----------------------------------------------------------------------------

void Dispatcher::run_stage5(Sequence *seq) {
  auto edge_count = cached_algorithms.at("edge_count").get();

  const int32_t n = app_data_ref.get_n_brt_nodes();

  edge_count->update_push_constants(EdgeCountPushConstants{
      .n_brt_nodes = n,
  });

  seq->record_commands(edge_count, n);
  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 6: Prefix Sum
// ----------------------------------------------------------------------------

void Dispatcher::run_stage6(Sequence *seq) {
  auto prefix_sum = cached_algorithms.at("prefix_sum").get();

  const uint32_t n = app_data_ref.get_n_brt_nodes();

  prefix_sum->update_descriptor_sets({
      engine_ref.get_buffer(app_data_ref.u_edge_count.data()),
      engine_ref.get_buffer(app_data_ref.u_edge_offset.data()),
  });

  prefix_sum->update_push_constants(PrefixSumPushConstants{
      .inputSize = n,
  });

  seq->record_commands_with_blocks(prefix_sum, 1);
  seq->launch_kernel_async();
  seq->sync();

  //   const auto n_octree_nodes =
  //   data.u_edge_offset->at(data.get_n_brt_nodes());

  //   spdlog::info("n_octree_nodes: {}", n_octree_nodes);
  //   data.set_n_octree_nodes(n_octree_nodes);

  const auto n_octree_nodes = app_data_ref.u_edge_offset[n - 1] + 1;
  app_data_ref.set_n_octree_nodes(n_octree_nodes);
}

// ----------------------------------------------------------------------------
// Stage 7: Build Octree
// ----------------------------------------------------------------------------

void Dispatcher::run_stage7(Sequence *seq) {
  auto build_octree = cached_algorithms.at("build_octree").get();

  const int32_t n = app_data_ref.get_n_brt_nodes();

  build_octree->update_push_constants(OctreePushConstants{
      .min_coord = app_data_ref.min_coord,
      .range = app_data_ref.range,
      .n_brt_nodes = n,
  });

  seq->record_commands(build_octree, n);
  seq->launch_kernel_async();
  seq->sync();
}

}  // namespace vulkan
