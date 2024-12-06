local headerfiles = {
  "*.hpp",
  "cuda/*.cuh",
}

local sources = {
  "host_kernels.cpp",
  "cuda/*.cu",
}

add_requires("cli11")

target("app-hello")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles(headerfiles)
    add_files("main.cpp", sources)
    add_packages("spdlog", "cli11")


    add_deps("cu-backend")
    
    -- tmp:
    add_deps("vk-backend")
    add_packages("vulkan-hpp", "vulkan-memory-allocator")


    add_cugencodes("native")



