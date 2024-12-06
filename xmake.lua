add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")

-- Global requirements
add_requires("spdlog")

includes("android.lua")

includes("redwood/cuda")
includes("redwood/vulkan")

includes("apps/hello_world")
