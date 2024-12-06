
add_requires("cli11")

target("app-hello")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles("*.hpp")
    add_files("main.cpp", "host_kernels.cpp")
    add_packages("spdlog", "cli11")

    -- CUDA related (optional)
    add_deps("cu-backend")
    add_headerfiles("cuda/*.cuh")
    add_files("cuda/*.cu")
    add_cugencodes("native")

    -- Android related (optional)
    if is_plat("android") then
      on_run(run_on_android)
    end
    add_deps("vk-backend")
    add_packages("vulkan-hpp", "vulkan-memory-allocator")





