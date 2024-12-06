#pragma once

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include "read_config.hpp"

// used for configuration
inline YAML::Node g_config;

// used for CPU thread pinning
inline std::vector<int> g_small_cores;
inline std::vector<int> g_medium_cores;
inline std::vector<int> g_big_cores;

inline void init_cores() {
  g_small_cores = helpers::get_cores_by_type(g_config["cpu_info"], "small");
  g_medium_cores = helpers::get_cores_by_type(g_config["cpu_info"], "medium");
  g_big_cores = helpers::get_cores_by_type(g_config["cpu_info"], "big");

  // all demos require at least one small core
  assert(!g_small_cores.empty());

  spdlog::info("Small cores: [{}]", fmt::join(g_small_cores, ", "));
  spdlog::info("Medium cores: [{}]", fmt::join(g_medium_cores, ", "));
  spdlog::info("Big cores: [{}]", fmt::join(g_big_cores, ", "));
}
