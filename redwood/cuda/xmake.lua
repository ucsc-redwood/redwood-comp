
option("cuda-backend")
    set_default(false)
    set_showmenu(true)
    set_description("Enable CUDA backend")
option_end()

if has_config("cuda-backend") then

set_policy("build.cuda.devlink", true)

target("cu-backend")
    set_kind("static")
    add_headerfiles("*.cuh")
    add_files("*.cu")
    add_packages("spdlog")
target_end()

end
