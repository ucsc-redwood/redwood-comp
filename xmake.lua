add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_toolchains("clang")
set_warnings("allextra")

-- Global requirements, for all projects
-- This is very handy library, I recommend it.
add_requires("spdlog")

includes("android.lua")

-- Backends
includes("redwood/host")
includes("redwood/cuda")
includes("redwood/vulkan")

includes("utility")
includes("apps")

-- includes("apps/playground")