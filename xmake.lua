add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")

option("cuda")
    set_default(false)
    set_showmenu(true)
    set_description("Enable CUDA support")
option_end()

-- Global requirements
add_requires("spdlog")

set_policy("build.cuda.devlink", true)

target("cu-backend")
    set_kind("static")
    add_headerfiles("redwood/cuda/*.cuh")
    add_files("redwood/cuda/*.cu")
    add_packages("spdlog")
target_end()

includes("apps/hello_world")
