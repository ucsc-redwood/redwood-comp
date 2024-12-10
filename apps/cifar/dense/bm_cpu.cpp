#include <benchmark/benchmark.h>

#include "../../app.hpp"
#include "../app_data.hpp"
#include "host/host_dispatcher.hpp"

// ----------------------------------------------------------------------------
// Fixtures
// ----------------------------------------------------------------------------

class CPU_Pinned : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    app_data = std::make_unique<AppData>(std::pmr::new_delete_resource());
  }

  void TearDown(benchmark::State&) override { app_data.reset(); }

  std::unique_ptr<AppData> app_data;
};

// ----------------------------------------------------------------------------
// Benchmarks
// ----------------------------------------------------------------------------

#define DEFINE_PINNED_BENCHMARK(NAME, CORE_TYPE)         \
  BENCHMARK_DEFINE_F(CPU_Pinned, NAME##_##CORE_TYPE)     \
  (benchmark::State & state) {                           \
    const auto n_threads = state.range(0);               \
    core::thread_pool pool(g_##CORE_TYPE##_cores, true); \
    for (auto _ : state) {                               \
      cpu::NAME(*app_data, pool, n_threads);             \
    }                                                    \
  }

DEFINE_PINNED_BENCHMARK(run_stage1_sync, small)
DEFINE_PINNED_BENCHMARK(run_stage1_sync, medium)
DEFINE_PINNED_BENCHMARK(run_stage1_sync, big)

DEFINE_PINNED_BENCHMARK(run_stage2_sync, small)
DEFINE_PINNED_BENCHMARK(run_stage2_sync, medium)
DEFINE_PINNED_BENCHMARK(run_stage2_sync, big)

DEFINE_PINNED_BENCHMARK(run_stage3_sync, small)
DEFINE_PINNED_BENCHMARK(run_stage3_sync, medium)
DEFINE_PINNED_BENCHMARK(run_stage3_sync, big)

DEFINE_PINNED_BENCHMARK(run_stage4_sync, small)
DEFINE_PINNED_BENCHMARK(run_stage4_sync, medium)
DEFINE_PINNED_BENCHMARK(run_stage4_sync, big)

DEFINE_PINNED_BENCHMARK(run_stage5_sync, small)
DEFINE_PINNED_BENCHMARK(run_stage5_sync, medium)
DEFINE_PINNED_BENCHMARK(run_stage5_sync, big)

DEFINE_PINNED_BENCHMARK(run_stage6_sync, small)
DEFINE_PINNED_BENCHMARK(run_stage6_sync, medium)
DEFINE_PINNED_BENCHMARK(run_stage6_sync, big)

DEFINE_PINNED_BENCHMARK(run_stage7_sync, small)
DEFINE_PINNED_BENCHMARK(run_stage7_sync, medium)
DEFINE_PINNED_BENCHMARK(run_stage7_sync, big)

DEFINE_PINNED_BENCHMARK(run_stage8_sync, small)
DEFINE_PINNED_BENCHMARK(run_stage8_sync, medium)
DEFINE_PINNED_BENCHMARK(run_stage8_sync, big)

DEFINE_PINNED_BENCHMARK(run_stage9_sync, small)
DEFINE_PINNED_BENCHMARK(run_stage9_sync, medium)
DEFINE_PINNED_BENCHMARK(run_stage9_sync, big)

#undef DEFINE_PINNED_BENCHMARK

void RegisterBenchmarkWithRange(const int n_small_cores,
                                const int n_medium_cores,
                                const int n_big_cores) {
#define REGISTER_BENCHMARK(NAME, CORE_TYPE, N_CORES)         \
  if (N_CORES > 0) {                                         \
    for (int i = 1; i <= N_CORES; ++i) {                     \
      ::benchmark::internal::RegisterBenchmarkInternal(      \
          new CPU_Pinned_##NAME##_##CORE_TYPE##_Benchmark()) \
          ->Arg(i)                                           \
          ->Name("CPU_Pinned/" #NAME "/" #CORE_TYPE)         \
          ->Unit(benchmark::kMillisecond)                    \
          ->Iterations(100);                                 \
    }                                                        \
  }

  REGISTER_BENCHMARK(run_stage1_sync, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage1_sync, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage1_sync, big, n_big_cores);

  REGISTER_BENCHMARK(run_stage2_sync, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage2_sync, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage2_sync, big, n_big_cores);

  REGISTER_BENCHMARK(run_stage3_sync, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage3_sync, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage3_sync, big, n_big_cores);

  REGISTER_BENCHMARK(run_stage4_sync, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage4_sync, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage4_sync, big, n_big_cores);

  REGISTER_BENCHMARK(run_stage5_sync, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage5_sync, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage5_sync, big, n_big_cores);

  REGISTER_BENCHMARK(run_stage6_sync, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage6_sync, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage6_sync, big, n_big_cores);

  REGISTER_BENCHMARK(run_stage7_sync, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage7_sync, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage7_sync, big, n_big_cores);

  REGISTER_BENCHMARK(run_stage8_sync, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage8_sync, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage8_sync, big, n_big_cores);

  REGISTER_BENCHMARK(run_stage9_sync, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage9_sync, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage9_sync, big, n_big_cores);

#undef REGISTER_BENCHMARK
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  INIT_APP("bm_cifar_dense");

  RegisterBenchmarkWithRange(
      g_small_cores.size(), g_medium_cores.size(), g_big_cores.size());

  // --------------------------------------------------------------------------
  // Run benchmarks
  // --------------------------------------------------------------------------

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return EXIT_SUCCESS;
}
