#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <memory>

#include "app_data.hpp"
#include "cuda/cu_dispatcher.cuh"
#include "redwood/cuda/cu_mem_resource.cuh"
#include "redwood/cuda/helpers.cuh"

// ----------------------------------------------------------------------------
// Fixtures
// ----------------------------------------------------------------------------

class iGPU_CUDA : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Keep the memory resource alive for the lifetime of the fixture
    mr = std::make_unique<cuda::CudaMemoryResource>();
    app_data = std::make_unique<AppData>(mr.get());
  }

  void TearDown(benchmark::State&) override {
    CUDA_CHECK(cudaStreamSynchronize(stream));

    app_data.reset();
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  std::unique_ptr<cuda::CudaMemoryResource> mr;
  std::unique_ptr<AppData> app_data;
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
  return 0;
}
