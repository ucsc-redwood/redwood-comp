#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <random>
#include <stdexcept>

#include "redwood/base_appdata.hpp"

// Use an educated guess for the memory ratio
// 60% memory is an empirical value
constexpr auto k_memory_ratio = 0.6f;

struct AppData final : public BaseAppData {
  static constexpr auto min_coord = 0.0f;
  static constexpr auto range = 1024.0f;

  explicit AppData(std::pmr::memory_resource* mr, const size_t n_input)
      : BaseAppData(mr),
        n_input(n_input),
        u_input_points(n_input, mr),
        u_morton_keys(n_input, mr),
        u_morton_keys_alt(n_input, mr),
        u_edge_count(n_input, mr),
        u_edge_offset(n_input, mr),
        brt(n_input, mr),
        oct(n_input * k_memory_ratio, mr)
  // u_contributes(n_input, mr),
  // u_out_idx(n_input, mr)
  {
    constexpr auto seed = 114514;

    std::mt19937 gen(seed);
    std::uniform_real_distribution dis(min_coord, min_coord + range);

    std::ranges::generate(u_input_points, [&]() {
      return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
    });
  }

  ~AppData() override = default;

  // --------------------------------------------------------------------------
  // Essential data
  // --------------------------------------------------------------------------

  const uint32_t n_input;
  uint32_t n_unique = std::numeric_limits<uint32_t>::max();
  uint32_t n_brt_nodes = std::numeric_limits<uint32_t>::max();
  uint32_t n_octree_nodes = std::numeric_limits<uint32_t>::max();

  // n_input
  UsmVector<glm::vec4> u_input_points;
  UsmVector<uint32_t> u_morton_keys;
  UsmVector<uint32_t> u_morton_keys_alt;

  // should have size 'n_brt_nodes', but for simplicity, we allocate the same
  // size as the input buffer 'n_input'
  UsmVector<int32_t> u_edge_count;
  UsmVector<int32_t> u_edge_offset;

  struct RadixTree {
    explicit RadixTree(const size_t n_nodes, std::pmr::memory_resource* mr)
        : u_prefix_n(n_nodes, mr),
          u_has_leaf_left(n_nodes, mr),
          u_has_leaf_right(n_nodes, mr),
          u_left_child(n_nodes, mr),
          u_parents(n_nodes, mr) {}

    UsmVector<uint8_t> u_prefix_n;
    UsmVector<uint8_t> u_has_leaf_left;
    UsmVector<uint8_t> u_has_leaf_right;
    UsmVector<int32_t> u_left_child;
    UsmVector<int32_t> u_parents;
  } brt;

  struct Octree {
    explicit Octree(const size_t n_nodes, std::pmr::memory_resource* mr)
        : u_children(n_nodes * 8, mr),
          u_corner(n_nodes, mr),
          u_cell_size(n_nodes, mr),
          u_child_node_mask(n_nodes, mr),
          u_child_leaf_mask(n_nodes, mr) {}

    // int (*u_children)[8]; note, this is 8x more
    UsmVector<int32_t> u_children;

    // everything else is size of 'n_octree_nodes'
    // but for simplicity, we allocate the 0.6 times of input size
    // 60% memory is an empirical value
    UsmVector<glm::vec4> u_corner;
    UsmVector<float> u_cell_size;
    UsmVector<int32_t> u_child_node_mask;
    UsmVector<int32_t> u_child_leaf_mask;
  } oct;

  // --------------------------------------------------------------------------
  // Intermediate data
  // --------------------------------------------------------------------------
  // UsmVector<uint32_t> u_contributes;
  // UsmVector<uint32_t> u_out_idx;

  // --------------------------------------------------------------------------
  // Getters
  // --------------------------------------------------------------------------

  [[nodiscard]] uint32_t get_n_input() const { return n_input; }

  [[nodiscard]] uint32_t get_n_unique() const {
    if (n_unique == std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("n_unique is not set");
    }

    return n_unique;
  }

  void set_n_unique(const uint32_t n_unique) { this->n_unique = n_unique; }
  void set_n_brt_nodes(const uint32_t n_brt_nodes) {
    this->n_brt_nodes = n_brt_nodes;
  }

  [[nodiscard]] uint32_t get_n_brt_nodes() const {
    return this->get_n_unique() - 1;
  }

  [[nodiscard]] uint32_t get_n_octree_nodes() const {
    if (n_octree_nodes == std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("n_octree_nodes is not set");
    }

    return n_octree_nodes;
  }

  void set_n_octree_nodes(const uint32_t n_octree_nodes) {
    this->n_octree_nodes = n_octree_nodes;
  }

  // The Idea is.

  // after calling sort, 'u_morton_keys_alt' is sorted
  // after calling move_dups, 'u_morton_keys' is sorted and unique
  // there's no way to check if you called, so I trust you call them in order

  [[nodiscard]] uint32_t* get_sorted_morton_keys() {
    return u_morton_keys_alt.data();
  }
  [[nodiscard]] const uint32_t* get_sorted_morton_keys() const {
    return u_morton_keys_alt.data();
  }

  [[nodiscard]] uint32_t* get_unique_morton_keys() {
    return u_morton_keys.data();
  }
  [[nodiscard]] const uint32_t* get_unique_morton_keys() const {
    return u_morton_keys.data();
  }
};
