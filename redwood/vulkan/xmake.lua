add_requires("vulkan-hpp", "vulkan-memory-allocator")

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
    add_packages("spdlog")
target_end()

end

