#pragma once

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <CLI/CLI.hpp>

#include "redwood/resources_path.hpp"

namespace helpers {

[[nodiscard]] inline YAML::Node load_config(const std::string& device_id) {
  const std::string config_path =
      (get_resource_base_path() / (device_id + ".yaml")).string();

  spdlog::info("Config path: {}", config_path);

  try {
    return YAML::LoadFile(config_path);
  } catch (const YAML::Exception& e) {
    spdlog::error("Error reading YAML file: {}", e.what());
    exit(1);
  }
}

[[nodiscard]] inline YAML::Node init_demo(int argc,
                                          char** argv,
                                          const std::string_view app_name) {
  std::string device_id;
  std::string log_level;

  CLI::App app{std::string(app_name)};
  app.add_option("-d,--device", device_id, "Device ID")->required();
  app.add_option("-l,--log-level", log_level, "Log level")->default_val("info");

  app.allow_extras();

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    spdlog::error("Error parsing CLI arguments: {}", e.what());
    std::cout << app.help() << std::endl;

    exit(e.get_exit_code());
  }

  spdlog::set_level(spdlog::level::from_str(log_level));

  return load_config(device_id);
}

}  // namespace helpers