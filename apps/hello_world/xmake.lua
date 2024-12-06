local headerfiles = {
  "*.hpp",
  -- "cuda/*.cuh",
}

local sources = {
  "host_kernels.cpp",
}

add_requires("cli11")

target("app-hello")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles(headerfiles)
    add_files("main.cpp", "host_kernels.cpp")
    add_packages("spdlog", "cli11")


    -- add_deps("cu-backend")
    -- add_cugencodes("native")

    if is_plat("android") then
      on_run(run_on_android)
    end

    -- tmp:
    add_deps("vk-backend")
    add_packages("vulkan-hpp", "vulkan-memory-allocator")





