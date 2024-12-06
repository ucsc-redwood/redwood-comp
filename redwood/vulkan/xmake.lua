add_requires("vulkan-hpp", "vulkan-memory-allocator")

target("vk-backend")
    set_kind("static")
    add_headerfiles("*.hpp")
    add_files("*.cpp")
    add_packages("vulkan-hpp", "vulkan-memory-allocator")
    add_packages("spdlog")
target_end()
