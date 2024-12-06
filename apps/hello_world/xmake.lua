local sources = {
  "host_kernels.cpp",
  "device_kernels.cu",
}

target("app-hello")
    set_kind("binary")
    add_includedirs("$(projectdir)")
    add_files("cuda.cpp", sources)
    add_packages("spdlog")
    add_deps("cu-backend")
    add_cugencodes("native")



