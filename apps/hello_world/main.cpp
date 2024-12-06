#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

// #include "redwood/cuda/cu_buffer.cuh"
// #include "redwood/host/host_buffer.hpp"
#include "redwood/cuda/cu_allocator.cuh"
#include "redwood/host/host_allocator.hpp"
#include "redwood/uni_engine.hpp"

// template <typename BufferT>
//   requires std::is_base_of_v<BaseBuffer, BufferT>

template <typename AllocatorT>
struct AppData {
  explicit AppData(const size_t n) : n(n), input_a(n), input_b(n), output(n) {
    std::ranges::fill(input_a, 1.0f);
    std::ranges::fill(input_b, 2.0f);
    std::ranges::fill(output, 0.0f);
  }

  const size_t n;
  std::vector<float, AllocatorT> input_a;
  std::vector<float, AllocatorT> input_b;
  std::vector<float, AllocatorT> output;

  // explicit AppData(UniEngine &eng, const size_t n) : n(n) {
  //   input_a = eng.buffer<BufferT>(n);
  //   input_b = eng.buffer<BufferT>(n);
  //   output = eng.buffer<BufferT>(n);
  // }

  // const size_t n;
  // std::shared_ptr<BufferT> input_a;
  // std::shared_ptr<BufferT> input_b;
  // std::shared_ptr<BufferT> output;
};

int main(int argc, char** argv) {
  CLI::App app("Hello World");

  bool use_cuda = false;
  app.add_flag("--cuda", use_cuda, "Use CUDA");
  CLI11_PARSE(app, argc, argv);

  spdlog::set_level(spdlog::level::trace);

  // Initialize compute engine and application data
  UniEngine engine;

  AppData<cpu::HostAllocator<float>> app_data(1024);
  AppData<cuda::CudaAllocator<float>> app_data_cuda(1024);

  // peek 10 elements
  for (size_t i = 0; i < 10; ++i) {
    spdlog::info("app_data[{}]: {}", i, app_data.input_a[i]);
  }

  return EXIT_SUCCESS;
}
