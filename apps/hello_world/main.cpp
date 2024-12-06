#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "redwood/cuda/cu_buffer.cuh"
#include "redwood/host/host_buffer.hpp"
#include "redwood/uni_engine.hpp"

template <typename BufferT>
  requires std::is_base_of_v<BaseBuffer, BufferT>
struct AppData {
  explicit AppData(UniEngine &eng, const size_t n) : n(n) {
    input_a = eng.buffer<BufferT>(n);
    input_b = eng.buffer<BufferT>(n);
    output = eng.buffer<BufferT>(n);
  }

  const size_t n;
  std::shared_ptr<BufferT> input_a;
  std::shared_ptr<BufferT> input_b;
  std::shared_ptr<BufferT> output;
};

int main(int argc, char **argv) {
  CLI::App app("Hello World");

  bool use_cuda = false;
  app.add_flag("--cuda", use_cuda, "Use CUDA");
  CLI11_PARSE(app, argc, argv);

  spdlog::set_level(spdlog::level::trace);

  // Initialize compute engine and application data
  UniEngine engine;

  if (use_cuda) {
    AppData<cuda::Buffer> app_data(engine, 1024);

  } else {
    AppData<cpu::HostBuffer> app_data(engine, 1024);
  }

  return EXIT_SUCCESS;
}
