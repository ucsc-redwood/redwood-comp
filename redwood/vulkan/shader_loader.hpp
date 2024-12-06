#pragma once

#include <string_view>

namespace vulkan {

/**
 * @brief Load GLSL source code from a file
 *
 * Reads a GLSL compute shader source file and returns its contents as a string.
 * The shader will be compiled to SPIR-V at runtime.
 *
 * @param filename Name of the GLSL source file
 * @return String containing the shader source code
 */
[[nodiscard]] const std::string load_source_from_file(
    const std::string_view filename);

}  // namespace vulkan
