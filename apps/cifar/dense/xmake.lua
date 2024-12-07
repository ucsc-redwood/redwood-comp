
target("app-cifar-dense")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles("*.hpp", "host/*.hpp")
    add_files("main.cpp", "host/*.cpp", "app_data.cpp")
    add_packages("spdlog", "cli11", "yaml-cpp")
    add_deps("cpu-backend")

    -- -- CUDA related (optional)
    -- if has_config("cuda-backend") then
    --   add_defines("REDWOOD_CUDA_BACKEND")
    --   add_deps("cu-backend")
    --   add_headerfiles("cuda/*.cuh")
    --   add_files("cuda/*.cu")
    --   add_cugencodes("native")
    -- end

    -- Android related (optional)
    if has_config("vulkan-backend") then
      add_defines("REDWOOD_VULKAN_BACKEND")
      add_headerfiles("vulkan/*.hpp")
      add_files("vulkan/*.cpp")

      if is_plat("android") then
        on_run(run_on_android)
      end
      
      add_deps("vk-backend")
      add_packages("vulkan-hpp", "vulkan-memory-allocator")
    end

