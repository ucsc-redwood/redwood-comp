#pragma once

#include <filesystem>

namespace helpers {

[[nodiscard]] inline std::filesystem::path get_resource_base_path() {
#if defined(__ANDROID__)
  return "/data/local/tmp/resources/";
#else
  // build
  // └── linux
  //     └── x86_64
  //         ├── debug
  //         │   ├── bm-cifar-dense
  // resources
  return std::filesystem::current_path()
             .parent_path()
             .parent_path()
             .parent_path()
             .parent_path() /
         "resources";
#endif
}

}  // namespace helpers
