#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <glm/glm.hpp>
#include <memory>

#include "app_data.hpp"
#include "cuda/cu_dispatcher.cuh"
// #include "host/host_dispatchers.hpp"
#include "redwood/cuda/cu_mem_resource.cuh"
#include "redwood/cuda/helpers.cuh"
// #include "redwood/host/thread_pool.hpp"

// ----------------------------------------------------------------------------
// Global Vars
// ----------------------------------------------------------------------------

constexpr auto kInputSize = 640 * 480;

// ----------------------------------------------------------------------------
// Fixtures
// ----------------------------------------------------------------------------

class iGPU_CUDA : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Keep the memory resource alive for the lifetime of the fixture
    mr = std::make_unique<cuda::CudaMemoryResource>();
    app_data = std::make_unique<AppData>(mr.get(), kInputSize);
    im_storage = std::make_unique<cuda::ImStorage>(kInputSize);

    // // need to run all stages so some we can have real data
    // std::vector<int> all_cores{1, 2, 3, 4, 5, 6};

    // const auto n_threads = all_cores.size();

    // core::thread_pool pool(all_cores, false);

    // cpu::run_stage1(*app_data, pool, n_threads);
    // cpu::run_stage2(*app_data, pool, n_threads);
    // cpu::run_stage3(*app_data, pool, n_threads);
    // cpu::run_stage4(*app_data, pool, n_threads);
    // cpu::run_stage5(*app_data, pool, n_threads);
    // cpu::run_stage6(*app_data, pool, n_threads);
    // cpu::run_stage7(*app_data, pool, n_threads);

    cuda::run_stage1(*app_data, stream);
    cuda::run_stage2(*app_data, *im_storage, stream);
    cuda::run_stage3(*app_data, *im_storage, stream);
    cuda::run_stage4(*app_data, stream);
    cuda::run_stage5(*app_data, stream);
    cuda::run_stage6(*app_data, stream);
    cuda::run_stage7(*app_data, stream);

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void TearDown(benchmark::State&) override {
    CUDA_CHECK(cudaDeviceSynchronize());

    app_data.reset();
    im_storage.reset();
    mr.reset();

    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  std::unique_ptr<cuda::CudaMemoryResource> mr;
  std::unique_ptr<AppData> app_data;
  std::unique_ptr<cuda::ImStorage> im_storage;
  cudaStream_t stream;
};

// ----------------------------------------------------------------------------
// Benchmarks
// ----------------------------------------------------------------------------

#define DEFINE_PINNED_BENCHMARK(NAME) \
  BENCHMARK_DEFINE_F(iGPU_CUDA, NAME) \
  (benchmark::State & state) {        \
    for (auto _ : state) {            \
      cuda::NAME(*app_data, stream);  \
    }                                 \
  }

#define DEFINE_PINNED_BENCHMARK_TMP(NAME)         \
  BENCHMARK_DEFINE_F(iGPU_CUDA, NAME)             \
  (benchmark::State & state) {                    \
    for (auto _ : state) {                        \
      cuda::NAME(*app_data, *im_storage, stream); \
    }                                             \
  }

DEFINE_PINNED_BENCHMARK(run_stage1)
DEFINE_PINNED_BENCHMARK_TMP(run_stage2)
DEFINE_PINNED_BENCHMARK_TMP(run_stage3)
DEFINE_PINNED_BENCHMARK(run_stage4)
DEFINE_PINNED_BENCHMARK(run_stage5)
DEFINE_PINNED_BENCHMARK(run_stage6)
DEFINE_PINNED_BENCHMARK(run_stage7)

#undef DEFINE_PINNED_BENCHMARK

void RegisterBenchmarks() {
#define REGISTER_BENCHMARK(NAME)                    \
  ::benchmark::internal::RegisterBenchmarkInternal( \
      new iGPU_CUDA_##NAME##_Benchmark())           \
      ->Name("iGPU_CUDA/" #NAME)                    \
      ->Unit(benchmark::kMillisecond)               \
      ->Iterations(100);

  REGISTER_BENCHMARK(run_stage1);
  REGISTER_BENCHMARK(run_stage2);
  REGISTER_BENCHMARK(run_stage3);
  REGISTER_BENCHMARK(run_stage4);
  REGISTER_BENCHMARK(run_stage5);
  REGISTER_BENCHMARK(run_stage6);
  REGISTER_BENCHMARK(run_stage7);

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
