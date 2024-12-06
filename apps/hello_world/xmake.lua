local headerfiles = {
  "*.hpp",
  "*.cuh",
}

local sources = {
  "host_kernels.cpp",
  "device_kernels.cu",
  "device_dispatchers.cu",
}

add_requires("cli11")

target("app-hello")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_headerfiles(headerfiles)
    add_files("cuda.cpp", sources)
    add_packages("spdlog", "cli11")
    add_deps("cu-backend")
    add_cugencodes("native")



