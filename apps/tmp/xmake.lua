target("vk_vma_pmr")
    set_kind("binary")
    add_files("*.cpp")
    add_packages("vulkan-hpp", "vulkan-memory-allocator")
    add_packages("shaderc", "spirv-reflect")
    add_packages("spdlog")
    add_links("vulkan")
    add_defines("SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE")
    if is_plat("android") then
        on_run(run_on_android)
    end
