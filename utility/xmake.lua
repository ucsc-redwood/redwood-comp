-- Utility targets that helps you know more about your system

target("check-affinity")
    set_kind("binary")
    add_files("check-affinity.cpp")
    if is_plat("android") then on_run(run_on_android) end
target_end()

target("print-core-info")
    set_kind("binary")
    add_files("print-core-info.cpp")
    if is_plat("android") then on_run(run_on_android) end
target_end()
