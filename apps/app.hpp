#pragma once

#include "cli_to_config.hpp"
#include "read_config.hpp"

inline YAML::Node g_config;

// used for CPU thread pinning
inline std::vector<int> g_small_cores;
inline std::vector<int> g_medium_cores;
inline std::vector<int> g_big_cores;

inline void init_app(int argc,
                     char** argv,
                     const std::string_view app_name = "default") {
  g_config = helpers::init_demo(argc, argv, app_name);

  g_small_cores = helpers::get_cores_by_type(g_config["cpu_info"], "small");
  g_medium_cores = helpers::get_cores_by_type(g_config["cpu_info"], "medium");
  g_big_cores = helpers::get_cores_by_type(g_config["cpu_info"], "big");

  // all demos require at least one small core
  assert(!g_small_cores.empty());
}

#define INIT_APP(demo_name) init_app(argc, argv, demo_name);
