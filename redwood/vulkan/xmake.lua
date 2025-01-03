-- add_requires("vulkan-hpp", "vulkan-memory-allocator")

add_requires("vulkan-hpp 1.3.290")
add_requires("vulkan-memory-allocator")

-- glsl compiler and reflection
add_requires("shaderc", "spirv-reflect")

option("vulkan-backend")
    set_default(false)
    set_showmenu(true)
    set_description("Enable Vulkan backend")
option_end()

if has_config("vulkan-backend") then

target("vk-backend")
    set_kind("static")
    add_headerfiles("*.hpp")
    add_files("*.cpp")
    add_packages("vulkan-hpp", "vulkan-memory-allocator")
    add_packages("shaderc", "spirv-reflect")
    add_packages("spdlog")
target_end()

end

