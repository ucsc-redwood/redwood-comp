
add_requires("cli11")

target("app-hello")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles("*.hpp", "host/*.hpp")
    add_files("main.cpp", "host/*.cpp")
    add_packages("spdlog", "cli11")

    -- CUDA related (optional)
    if has_config("cuda-backend") then
      add_defines("REDWOOD_CUDA_BACKEND")
      add_deps("cu-backend")
      add_headerfiles("cuda/*.cuh")
      add_files("cuda/*.cu")
      add_cugencodes("native")
    end

    -- Android related (optional)
    if has_config("vulkan-backend") then
      add_defines("REDWOOD_VULKAN_BACKEND")

      if is_plat("android") then
        on_run(run_on_android)
      end
      
      add_deps("vk-backend")
      add_packages("vulkan-hpp", "vulkan-memory-allocator")
    end




