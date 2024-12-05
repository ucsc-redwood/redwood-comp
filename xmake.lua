add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")

-- Global requirements
add_requires("spdlog")

add_repositories("local build")

-- Two exclusive backends: CUDA and Vulkan(Android), and a CPU backend

add_requires("vulkan-hpp", "vulkan-memory-allocator")
add_requires("shaderc", "spirv-reflect")

add_requires("redwood-comp-cu")

target("redwood-comp-vk")
    set_kind("static")
    add_headerfiles("src/redwood/vulkan/*.hpp")
    add_files("src/redwood/vulkan/*.cpp")
    add_packages("vulkan-hpp", "vulkan-memory-allocator")
    add_packages("shaderc", "spirv-reflect")
    add_packages("spdlog")


target("redwood-comp-cu")
    set_kind("static")
    add_headerfiles("src/redwood/cuda/*.cuh")
    add_files("src/redwood/cuda/*.cu")
    add_cugencodes("native")
    add_packages("spdlog")
    add_links("cudart")
    add_cuflags("-rdc=true")


target("demo-cu")
    set_kind("binary")
    add_includedirs("src/redwood/")
    add_files("demo/demo_cuda.cpp")
    add_packages("spdlog")
    add_deps("redwood-comp-cu")
    add_links("cudart")
    add_linkdirs("/opt/cuda/lib64")

-- target("demo-vk")
--     set_kind("binary")
--     add_includedirs("src/redwood/")
--     add_files("demo/demo_vk.cpp") 
--     add_packages("vulkan-hpp", "vulkan-memory-allocator")
--     -- add_packages("shaderc", "spirv-reflect")
--     add_packages("spdlog")
--     add_deps("redwood-comp-vk")
--     add_deps("redwood-comp-cu")
