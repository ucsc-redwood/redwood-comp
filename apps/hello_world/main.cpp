#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <memory_resource>

#include "app_data.hpp"
#include "host/host_dispatchers.hpp"
#include "redwood/backends.hpp"

// forward declare
void run_cpu_backend_demo(size_t n);
void run_cuda_backend_demo(size_t n);
void run_vulkan_backend_demo(size_t n);

// print the first 10 elements of the output
void print_output(const AppData& app_data) {
  for (size_t i = 0; i < 10; ++i) {
    spdlog::info("output[{}] = {}", i, app_data.u_output[i]);
  }
}

#ifdef REDWOOD_CUDA_BACKEND

// #include "cuda/device_dispatchers.cuh"
#include "cuda/device_kernels.cuh"
#include "redwood/cuda/cu_dispatcher.cuh"
#include "redwood/cuda/cu_mem_resource.cuh"
#include "redwood/cuda/helpers.cuh"

void run_cuda_backend_demo(const size_t n) {
  auto mr = std::make_shared<cuda::CudaMemoryResource>();

  AppData app_data(mr.get(), n);

  // cuda::run_stage1(app_data);

  cuda::CuDispatcher dispatcher(mr, 1);
  constexpr size_t my_queue_idx = 0;

  dispatcher.dispatch(my_queue_idx, [&](const size_t stream_id) {
    constexpr auto threads = 256;
    const auto blocks = div_up(app_data.n, threads);
    constexpr auto s_mem = 0;

    auto start = 0;
    auto end = app_data.n;
    void* args[] = {app_data.u_input_a.data(),
                    app_data.u_input_b.data(),
                    app_data.u_output.data(),
                    &start,
                    &end};

    cudaLaunchKernel(
        (void*)cuda::kernels::vector_add,  // Function pointer to the kernel
        dim3(blocks),                      // Grid dimensions
        dim3(threads),                     // Block dimensions
        args,                              // Arguments to the kernel
        s_mem,                             // Shared memory size
        dispatcher.stream(stream_id)       // Stream
    );

    // cuda::kernels::
    //     vector_add<<<blocks, threads, s_mem, dispatcher.stream(stream_id)>>>(
    //         app_data.u_input_a.data(),
    //         app_data.u_input_b.data(),
    //         app_data.u_output.data(),
    //         0,
    //         app_data.n);
  });

  CUDA_CHECK(cudaDeviceSynchronize());

  // dispatcher.synchronize(my_queue_idx);
  print_output(app_data);
}

#endif

#ifdef REDWOOD_VULKAN_BACKEND

#include "redwood/vulkan/vk_allocator.hpp"
#include "vulkan/vk_dispatchers.hpp"

void run_vulkan_backend_demo(const size_t n) {
  vulkan::Engine engine;
  vulkan::VulkanMemoryResource vk_mr(engine);
  AppData app_data(&vk_mr, n);

  vulkan::run_stage1(engine, app_data);
  print_output(app_data);
}

#endif

void run_cpu_backend_demo(const size_t n) {
  auto host_mr = std::pmr::new_delete_resource();
  AppData app_data(host_mr, n);

  cpu::run_stage1(app_data).wait();
  print_output(app_data);
}

int main(int argc, char** argv) {
  CLI::App app("Hello World");

  std::string device_id;
  app.add_option("-d,--device", device_id, "Device ID")->required();
  CLI11_PARSE(app, argc, argv);

  spdlog::set_level(spdlog::level::trace);

  constexpr auto n = 1024;

  if constexpr (is_backend_enabled(BackendType::kCPU)) {
    spdlog::info("CPU backend is enabled");
    run_cpu_backend_demo(n);
  }
  if constexpr (is_backend_enabled(BackendType::kCUDA)) {
    spdlog::info("CUDA backend is enabled");
    run_cuda_backend_demo(n);
  }
  if constexpr (is_backend_enabled(BackendType::kVulkan)) {
    spdlog::info("Vulkan backend is enabled");
    run_vulkan_backend_demo(n);
  }

  spdlog::info("Done");
  return EXIT_SUCCESS;
}
