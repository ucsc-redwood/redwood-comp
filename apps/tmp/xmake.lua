
target("vk_vma_pmr")
    set_kind("binary")
    add_files("*.cpp")
    add_includedirs("$(projectdir)/")

    add_packages("vulkan-hpp")
    add_packages("vulkan-memory-allocator")

    add_packages("shaderc", "spirv-reflect")
    add_packages("spdlog")
    add_defines("SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE")
    if is_plat("android") then
        on_run(run_on_android)
    end
