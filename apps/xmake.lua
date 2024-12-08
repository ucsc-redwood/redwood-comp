if is_plat("android") then
    package("benchmark")
        set_kind("library")
        add_deps("cmake")
        set_urls("https://github.com/google/benchmark.git")
        add_versions("v1.9.0", "12235e24652fc7f809373e7c11a5f73c5763fc4c")
        
        -- Add description and homepage for better package management
        set_description("A microbenchmark support library")
        set_homepage("https://github.com/google/benchmark")

        on_install(function(package)
            local configs = {
                "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"),
                "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"),
                "-DBENCHMARK_DOWNLOAD_DEPENDENCIES=on",
                "-DHAVE_THREAD_SAFETY_ATTRIBUTES=0"
            }
            import("package.tools.cmake").install(package, configs)
        end)
    package_end()
end

add_requires("yaml-cpp")
add_requires("benchmark")
add_requires("cli11")

includes("hello_world")
includes("cifar/dense")
-- includes("tmp")

