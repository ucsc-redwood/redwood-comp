add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")

-- Global requirements
add_requires("spdlog")

set_policy("build.cuda.devlink", true)

target("cu-backend")
    set_kind("static")
    add_headerfiles("app/cuda/*.cuh")
    add_files("app/cuda/*.cu")
    add_packages("spdlog")

target("app")
    set_kind("binary")
    add_files("app/main.cpp")
    add_packages("spdlog")
    add_deps("cu-backend")
