#pragma once

#include <yaml-cpp/yaml.h>

namespace helpers {

[[nodiscard]] inline std::vector<int> get_cores_by_type(
    const YAML::Node& cpu_info,
    const std::string& core_type,
    const bool only_pinnable = true) {
  std::vector<int> result;

  if (!cpu_info["cores"]) {
    return result;
  }

  for (const auto& core : cpu_info["cores"]) {
    if (core["type"].as<std::string>() == core_type &&
        (!only_pinnable || core["pinnable"].as<bool>())) {
      result.push_back(core["id"].as<int>());
    }
  }

  return result;
}

// Function to get all unique core types
[[nodiscard]] inline std::vector<std::string> get_core_types(
    const YAML::Node& cpu_info) {
  std::vector<std::string> types;
  std::unordered_map<std::string, bool> type_map;

  if (!cpu_info["cores"]) {
    return types;
  }

  for (const auto& core : cpu_info["cores"]) {
    std::string type = core["type"].as<std::string>();
    if (!type_map[type]) {
      type_map[type] = true;
      types.push_back(type);
    }
  }

  return types;
}

}  // namespace helpers