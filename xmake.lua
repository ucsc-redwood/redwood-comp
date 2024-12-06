add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")

-- Global requirements
add_requires("spdlog")

set_policy("build.cuda.devlink", true)

target("cu-backend")
    set_kind("static")
    add_headerfiles("redwood/cuda/*.cuh")
    add_files("redwood/cuda/*.cu")
    add_packages("spdlog")

includes("apps/hello_world")