#include <benchmark/benchmark.h>

#include "../app.hpp"
#include "app_data.hpp"
#include "host/host_dispatchers.hpp"

// ----------------------------------------------------------------------------
// Global Vars
// ----------------------------------------------------------------------------
constexpr auto kInputSize = 640 * 480;

// ----------------------------------------------------------------------------
// Fixtures
// ----------------------------------------------------------------------------

class CPU_Pinned : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    app_data =
        std::make_unique<AppData>(std::pmr::new_delete_resource(), kInputSize);

    // need to run all stages so some we can have real data
    std::vector<int> all_cores;
    all_cores.reserve(g_small_cores.size() + g_medium_cores.size() +
                      g_big_cores.size());
    all_cores.insert(
        all_cores.end(), g_small_cores.begin(), g_small_cores.end());
    all_cores.insert(
        all_cores.end(), g_medium_cores.begin(), g_medium_cores.end());
    all_cores.insert(all_cores.end(), g_big_cores.begin(), g_big_cores.end());
    const auto n_threads = all_cores.size();

    // unpinned
    core::thread_pool pool(all_cores, false);

    cpu::run_stage1(*app_data, pool, n_threads);
    cpu::run_stage2(*app_data, pool, n_threads);
    cpu::run_stage3(*app_data, pool, n_threads);
    cpu::run_stage4(*app_data, pool, n_threads);
    cpu::run_stage5(*app_data, pool, n_threads);
    cpu::run_stage6(*app_data, pool, n_threads);
    cpu::run_stage7(*app_data, pool, n_threads);
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

DEFINE_PINNED_BENCHMARK(run_stage1, small)
DEFINE_PINNED_BENCHMARK(run_stage1, medium)
DEFINE_PINNED_BENCHMARK(run_stage1, big)

DEFINE_PINNED_BENCHMARK(run_stage2, small)
DEFINE_PINNED_BENCHMARK(run_stage2, medium)
DEFINE_PINNED_BENCHMARK(run_stage2, big)

DEFINE_PINNED_BENCHMARK(run_stage3, small)
DEFINE_PINNED_BENCHMARK(run_stage3, medium)
DEFINE_PINNED_BENCHMARK(run_stage3, big)

DEFINE_PINNED_BENCHMARK(run_stage4, small)
DEFINE_PINNED_BENCHMARK(run_stage4, medium)
DEFINE_PINNED_BENCHMARK(run_stage4, big)

DEFINE_PINNED_BENCHMARK(run_stage5, small)
DEFINE_PINNED_BENCHMARK(run_stage5, medium)
DEFINE_PINNED_BENCHMARK(run_stage5, big)

DEFINE_PINNED_BENCHMARK(run_stage6, small)
DEFINE_PINNED_BENCHMARK(run_stage6, medium)
DEFINE_PINNED_BENCHMARK(run_stage6, big)

DEFINE_PINNED_BENCHMARK(run_stage7, small)
DEFINE_PINNED_BENCHMARK(run_stage7, medium)
DEFINE_PINNED_BENCHMARK(run_stage7, big)

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

  REGISTER_BENCHMARK(run_stage1, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage1, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage1, big, n_big_cores);

  REGISTER_BENCHMARK(run_stage2, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage2, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage2, big, n_big_cores);

  REGISTER_BENCHMARK(run_stage3, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage3, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage3, big, n_big_cores);

  REGISTER_BENCHMARK(run_stage4, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage4, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage4, big, n_big_cores);

  REGISTER_BENCHMARK(run_stage5, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage5, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage5, big, n_big_cores);

  REGISTER_BENCHMARK(run_stage6, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage6, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage6, big, n_big_cores);

  REGISTER_BENCHMARK(run_stage7, small, n_small_cores);
  REGISTER_BENCHMARK(run_stage7, medium, n_medium_cores);
  REGISTER_BENCHMARK(run_stage7, big, n_big_cores);

#undef REGISTER_BENCHMARK
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  INIT_APP("bm_tree");

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
