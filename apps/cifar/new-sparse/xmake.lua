local cpp_source = {
  "host/*.cpp",
  "app_data.cpp"
}

local cpp_header = {
  "./*.hpp",
  "host/*.hpp"
}

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
