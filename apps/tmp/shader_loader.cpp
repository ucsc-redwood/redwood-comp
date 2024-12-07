#include "shader_loader.hpp"

#include <spdlog/spdlog.h>

#include <fstream>

// TODO: change
#include "redwood/resources_path.hpp"

namespace fs = std::filesystem;

namespace vulkan {

/**
 * @brief Loads a shader source file and returns its contents as a string
 *
 * @param filepath The path to the shader source file to load
 * @return std::string The shader source contents
 * @throws std::runtime_error if:
 *   - The shader file is not found
 *   - The file cannot be opened
 *   - There are errors reading the file
 */
[[nodiscard]] const std::string load_source_from_file(
    const std::string_view filename) {
  const fs::path shader_path =
      helpers::get_resource_base_path() / "shaders" / filename;

  spdlog::info("loading shader source: {}", shader_path.string());

  if (!fs::exists(shader_path)) {
    throw std::runtime_error("Shader file not found: " + shader_path.string());
  }

  std::ifstream file(shader_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + shader_path.string());
  }

  std::string source{std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>()};

  file.close();

  if (file.fail()) {
    throw std::runtime_error("Failed to read file: " + shader_path.string());
  }

  return source;
}

}  // namespace vulkan
