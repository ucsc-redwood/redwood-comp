set_policy("build.cuda.devlink", true)

target("cu-backend")
    set_kind("static")
    add_headerfiles("*.cuh")
    add_files("*.cu")
    add_packages("spdlog")
target_end()
