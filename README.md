# Unified Frontend


## Applicatioon

### Initialization

an application is need to pick a backend first.

you can create a backend by 'Engine' class.

```cpp
int main() {
  auto engine = Engine::create(EngineType::CUDA);
  // ...
}
```

### Define Application Specific Data 

Then you will allocate the memories you need for your specific application.

```cpp
auto input_a = engine->typed_buffer<int>(1024);
auto input_b = engine->typed_buffer<int>(1024);
auto output = engine->typed_buffer<int>(1024);
// ...
```

It is preferred to define ALL the data in `AppData` class.


```cpp
template <typename T>
using CudaBuffer = std::shared_ptr<cuda::TypedBuffer<T>>;
```

Define the data in `AppData` class.

```cpp
namespace app1 {


class AppData {

public:
  AppData(Engine& engine)
      : input_a(engine.typed_buffer<int>(1024)),
        input_b(engine.typed_buffer<int>(1024)),
        output(engine.typed_buffer<int>(1024)) {}

private:
  CudaBuffer<int> input_a;
  CudaBuffer<int> input_b;
  CudaBuffer<int> output;
};

}
```

### Define Application Specific Kernels

A kernel is a function that takes some data as input and produce some data as output. The kernel can be ran on either CPU or GPU.

This is an example of a kernel that you usually write.The kenel handles all the logic of your application. And you will need to write a for loop that iterates over the data to process each element.

```cpp
// your normal function 
void kernel_cpu(const int* input_a, 
                const int* input_b, 
                int* output, 
                const size_t size) {
  for (size_t i = 0; i < size; ++i) {
    // ...
  }
}
```

#### For CPU

just write normal functions. It is preferred to have the kernel taking raw pointers as input and output because it is more flexible for CUDA backend. 


However, for CPU kernels, we are going to parallelize the kernel execution using `pthread` (for fine control of parallelism). 

Thus it is *required* to define two more arguments: `start` and `end`, which indicates the range of the data that the thread is going to process.

```cpp
// modified version of the kernel
void kernel_cpu(const int* input_a, 
                const int* input_b, 
                int* output, 
                size_t size,
                // added arguments for parallelization
                const size_t start,
                const size_t end) {
  // iterate from 'start' to 'end'
  for (size_t i = start; i < end; ++i) {
    // ...
  }
}
```

#### For CUDA

#### For Vulkan

Need to write shader code. Will be compiled into SPIR-V.



## Organization of the Code

```cpp

// Define
struct AppData {
  CudaBuffer<int> input_a;
  CudaBuffer<int> input_b;
  CudaBuffer<int> output;
};

// backend specific namespace
namespace cpu {

  // kernels
  namespace kernel {
    void foo(int* output, int start, int end); 
    void bar(int* output, int start, int end);
  }

  // dispatchers
  std::future<void> run_stage1(AppData& app_data);
  std::future<void> run_stage2(AppData& app_data);
}

// in 'cuda/kernels/foo.cuh'
namespace cuda {

  namespace kernel {
    __global__ void foo(int* output, int n); 
    __global__ void bar(int* output, int n);
  }

  // dispatchers
  std::future<void> run_stage1(AppData& app_data);
  std::future<void> run_stage2(AppData& app_data);
}

```
