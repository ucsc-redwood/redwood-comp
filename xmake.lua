add_rules("mode.debug", "mode.release")

set_languages("c++20")
if not is_plat("android") then
    set_toolchains("clang")
end
set_warnings("allextra")

-- Global requirements, for all projects
-- This is very handy library, I recommend it.
add_requires("spdlog")

-- add_cuflags("-allow-unsupported-compiler", {force = true})

includes("android.lua")

-- Backends
includes("redwood/host")
includes("redwood/cuda")
includes("redwood/vulkan")

includes("utility")
includes("apps")

