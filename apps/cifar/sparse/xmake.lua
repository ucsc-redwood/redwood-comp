local cpp_source = {
  -- "host/*.cpp",
  "../app_data.cpp",
  "csr.cpp",
  "tmp/original_kernels.cpp",
  "sparse_app_data.cpp"
}

local cpp_header = {
  "../*.hpp",
  -- "host/*.hpp"
}

target("app-cifar-sparse")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles(cpp_header)
    add_files("main.cpp", cpp_source)
    add_packages("spdlog", "cli11", "yaml-cpp")
    add_deps("cpu-backend")
    if is_plat("android") then
      on_run(run_on_android)
    end

