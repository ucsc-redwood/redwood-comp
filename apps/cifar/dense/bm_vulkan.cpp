#include <benchmark/benchmark.h>

#include "../app_data.hpp"
#include "vulkan/vk_dispatcher.hpp"

// ----------------------------------------------------------------------------
// Fixtures
// ----------------------------------------------------------------------------

class iGPU_Vulkan : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    engine = std::make_unique<vulkan::Engine>();
    app_data = std::make_unique<AppData>(engine->get_mr());
    dispatcher = std::make_unique<vulkan::Dispatcher>(*engine, *app_data);
  }

  void TearDown(benchmark::State&) override {
    app_data.reset();
    engine.reset();
    dispatcher.reset();
  }

  std::unique_ptr<vulkan::Engine> engine;
  std::unique_ptr<AppData> app_data;
  std::unique_ptr<vulkan::Dispatcher> dispatcher;
};

// ----------------------------------------------------------------------------
// Benchmarks
// ----------------------------------------------------------------------------

#define DEFINE_PINNED_BENCHMARK(NAME)   \
  BENCHMARK_DEFINE_F(iGPU_Vulkan, NAME) \
  (benchmark::State & state) {          \
    auto seq = engine->sequence();      \
    for (auto _ : state) {              \
      dispatcher->NAME(seq.get());      \
    }                                   \
  }

DEFINE_PINNED_BENCHMARK(run_stage1)
DEFINE_PINNED_BENCHMARK(run_stage2)
DEFINE_PINNED_BENCHMARK(run_stage3)
DEFINE_PINNED_BENCHMARK(run_stage4)
DEFINE_PINNED_BENCHMARK(run_stage5)
DEFINE_PINNED_BENCHMARK(run_stage6)
DEFINE_PINNED_BENCHMARK(run_stage7)
DEFINE_PINNED_BENCHMARK(run_stage8)
DEFINE_PINNED_BENCHMARK(run_stage9)

#undef DEFINE_PINNED_BENCHMARK

void RegisterBenchmarks() {
#define REGISTER_BENCHMARK(NAME)                    \
  ::benchmark::internal::RegisterBenchmarkInternal( \
      new iGPU_Vulkan_##NAME##_Benchmark())         \
      ->Name("iGPU_Vulkan/" #NAME)                  \
      ->Unit(benchmark::kMillisecond)               \
      ->Iterations(100);

  REGISTER_BENCHMARK(run_stage1);
  REGISTER_BENCHMARK(run_stage2);
  REGISTER_BENCHMARK(run_stage3);
  REGISTER_BENCHMARK(run_stage4);
  REGISTER_BENCHMARK(run_stage5);
  REGISTER_BENCHMARK(run_stage6);
  REGISTER_BENCHMARK(run_stage7);
  REGISTER_BENCHMARK(run_stage8);
  REGISTER_BENCHMARK(run_stage9);

#undef REGISTER_BENCHMARK
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::off);

  RegisterBenchmarks();

  // --------------------------------------------------------------------------
  // Run benchmarks
  // --------------------------------------------------------------------------

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return EXIT_SUCCESS;
}
