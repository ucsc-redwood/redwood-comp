local cpp_source = {
  "host/*.cpp",
  "app_data.cpp"
}

local cpp_header = {
  "./*.hpp",
  "host/*.hpp"
}

-- ----------------------------------------------------------------------------
-- Application
-- ----------------------------------------------------------------------------

target("app-cifar-sparse-new")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles(cpp_header)
    add_files("main.cpp", cpp_source)
    add_packages("spdlog", "cli11", "yaml-cpp")
    add_deps("cpu-backend")
    if is_plat("android") then
      on_run(run_on_android)
    end

    -- CUDA related (optional)
    if has_config("cuda-backend") then
      add_defines("REDWOOD_CUDA_BACKEND")
      add_deps("cu-backend")
      add_headerfiles("cuda/*.cuh")
      add_files("cuda/*.cu")
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
target_end()

-- ----------------------------------------------------------------------------
-- Host Benchmark
-- ----------------------------------------------------------------------------

target("bm-cifar-sparse-cpu")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles(cpp_header)
    add_files("bm_cpu.cpp", cpp_source)
    add_packages("spdlog", "cli11", "yaml-cpp")
    add_deps("cpu-backend")
    if is_plat("android") then
      on_run(run_on_android)
    end

    add_packages("benchmark")

-- ----------------------------------------------------------------------------
-- Vulkan Benchmark
-- ----------------------------------------------------------------------------

if has_config("vulkan-backend") then

target("bm-cifar-sparse-vk")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles(cpp_header)
    add_files("bm_vulkan.cpp", cpp_source)
    add_packages("spdlog", "cli11", "yaml-cpp")

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


-- ----------------------------------------------------------------------------
-- CUDA Benchmark
-- ----------------------------------------------------------------------------

if has_config("cuda-backend") then

target("bm-cifar-sparse-cuda")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles(cpp_header)
    add_files("bm_cuda.cpp", cpp_source)
    add_packages("spdlog", "cli11", "yaml-cpp")

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
