
add_requires("glm")

local cu_source_files = {
  "cuda/01_morton.cu",
  "cuda/02_sort.cu",
  "cuda/03_unique.cu",
  "cuda/04_radix_tree.cu",
  "cuda/05_edge_count.cu",
  "cuda/06_prefix_sum.cu",
  "cuda/07_octree.cu",
  "cuda/cu_dispatcher.cu",
  "cuda/im_storage.cu",
}

target("app-tree")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles("*.hpp", "host/*.hpp")
    add_files("main.cpp", "host/*.cpp")
    add_packages("spdlog", "cli11", "yaml-cpp")
    add_packages("glm")
    add_deps("cpu-backend")
    if is_plat("android") then
      on_run(run_on_android)
    end

    -- CUDA related (optional)
    if has_config("cuda-backend") then
      add_defines("REDWOOD_CUDA_BACKEND")
      add_deps("cu-backend")
      add_headerfiles("cuda/**/*.cuh")
      add_files(cu_source_files)
      add_cugencodes("native")
    end

    -- Vulkan related (optional)
    if has_config("vulkan-backend") then
      add_defines("REDWOOD_VULKAN_BACKEND")
      add_headerfiles("vulkan/*.hpp")
      add_files("vulkan/*.cpp")      
      add_deps("vk-backend")
      add_packages("vulkan-hpp", "vulkan-memory-allocator")
    end

target("bm-tree-cpu")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles("*.hpp", "host/*.hpp")
    add_files("bm_cpu.cpp", "host/*.cpp")
    add_packages("spdlog", "cli11", "yaml-cpp")
    add_packages("glm")
    add_deps("cpu-backend")
    if is_plat("android") then
      on_run(run_on_android)
    end

    add_packages("benchmark")

if has_config("vulkan-backend") then

target("bm-tree-vk")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles("*.hpp")
    add_files("bm_vulkan.cpp")
    add_packages("spdlog", "cli11", "yaml-cpp")
    add_packages("glm")

    if is_plat("android") then
      on_run(run_on_android)
    end

    add_packages("benchmark")

    add_defines("REDWOOD_VULKAN_BACKEND")
    add_headerfiles("vulkan/*.hpp")
    add_files("vulkan/*.cpp")      
    add_deps("vk-backend")
    add_packages("vulkan-hpp", "vulkan-memory-allocator")
target_end()

end

if has_config("cuda-backend") then

target("bm-tree-cuda")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles("*.hpp")
    add_files("bm_cuda.cpp")
    add_packages("spdlog", "cli11", "yaml-cpp")

    add_packages("glm")

    if is_plat("android") then
      on_run(run_on_android)
    end

    add_packages("benchmark")
    
    add_defines("REDWOOD_CUDA_BACKEND")
    add_deps("cu-backend")
    add_headerfiles("cuda/*.cuh")
    add_files("cuda/*.cu")
    add_cugencodes("native")
target_end()

end



target("pipe-tree-baseline")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles("*.hpp", "host/*.hpp")
    add_files("pipe_main_baseline.cpp", "host/*.cpp")
    add_packages("spdlog", "cli11", "yaml-cpp")
    add_packages("glm")
    add_deps("cpu-backend")
    if is_plat("android") then
      on_run(run_on_android)
    end

    -- CUDA related (optional)
    if has_config("cuda-backend") then
      add_defines("REDWOOD_CUDA_BACKEND")
      add_deps("cu-backend")
      add_headerfiles("cuda/**/*.cuh")
      add_files(cu_source_files)
      add_cugencodes("native")
    end

    -- Vulkan related (optional)
    if has_config("vulkan-backend") then
      add_defines("REDWOOD_VULKAN_BACKEND")
      add_headerfiles("vulkan/*.hpp")
      add_files("vulkan/*.cpp")      
      add_deps("vk-backend")
      add_packages("vulkan-hpp", "vulkan-memory-allocator")
    end