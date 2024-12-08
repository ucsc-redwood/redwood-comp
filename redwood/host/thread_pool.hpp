#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace core {

inline void set_cpu_affinity(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
    throw std::system_error(
        errno,
        std::system_category(),
        "Failed to set CPU affinity to core " + std::to_string(core_id));
  }
}

template <typename T>
struct multi_future {
  std::vector<std::future<T>> futures;

  void add(std::future<T> &&fut) { futures.push_back(std::move(fut)); }

  // Wait for all futures in the collection
  void wait() {
    for (auto &fut : futures) {
      fut.wait();
    }
  }
};

class thread_pool {
 public:
  explicit thread_pool(int n_threads) : stopFlag(false) {
    workers.reserve(n_threads);

    for (int i = 0; i < n_threads; ++i) {
      workers.emplace_back([this] {
        while (true) {
          std::function<void()> task;

          // Lock the queue to retrieve tasks safely
          {
            std::unique_lock lock(queueMutex);
            condition.wait(lock, [this] { return stopFlag || !tasks.empty(); });

            if (stopFlag && tasks.empty()) return;

            task = std::move(tasks.front());
            tasks.pop();
          }

          // Execute the retrieved task outside the lock
          task();
        }
      });
    }
  }

  explicit thread_pool(std::vector<int> core_ids, bool enable_pinning = false)
      : stopFlag(false) {
    workers.reserve(core_ids.size());
    for (auto id : core_ids) {
      workers.emplace_back([this, id, enable_pinning] {
        // Pin the thread to the specified core only if enabled
        if (enable_pinning) {
          set_cpu_affinity(id);
        }

        while (true) {
          std::function<void()> task;

          // Lock the queue to retrieve tasks safely
          {
            std::unique_lock lock(queueMutex);
            condition.wait(lock, [this] { return stopFlag || !tasks.empty(); });

            if (stopFlag && tasks.empty()) return;

            task = std::move(tasks.front());
            tasks.pop();
          }

          // Execute the retrieved task outside the lock
          task();
        }
      });
    }
  }

  ~thread_pool() {
    {
      std::unique_lock lock(queueMutex);
      stopFlag = true;
    }
    condition.notify_all();

    for (std::thread &worker : workers)
      if (worker.joinable()) worker.join();
  }

  [[nodiscard]] size_t get_thread_count() const { return workers.size(); }

  template <class F, class... Args>
  auto submit_task(F &&f, Args &&...args)
      -> std::future<typename std::invoke_result_t<F, Args...>> {
    using ReturnType = typename std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<ReturnType> future = task->get_future();

    {
      std::lock_guard<std::mutex> lock(queueMutex);
      if (stopFlag)
        throw std::runtime_error("submit_task on stopped ThreadPool");

      tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();

    return future;
  }

  // No thread id
  template <typename T,
            typename F,
            typename R = std::invoke_result_t<std::decay_t<F>, T, T>>
  [[nodiscard]] multi_future<R> submit_blocks(const T first_index,
                                              const T index_after_last,
                                              F &&block,
                                              const size_t num_blocks = 0) {
    multi_future<R> future_collection;

    if (index_after_last > first_index) {
      size_t M = num_blocks ? num_blocks : workers.size();
      T block_size = (index_after_last - first_index + M - 1) / M;  // Round up

      for (size_t i = 0; i < M; ++i) {
        T start = first_index + i * block_size;
        T end = std::min(start + block_size, index_after_last);

        if (start >= index_after_last) break;

        // Submit each block as a separate task and add the future to
        // multi_future
        future_collection.add(
            submit_task([block = std::forward<F>(block), start, end] {
              return block(start, end);
            }));
      }
    }
    return future_collection;
  }

  // // With thread id
  // template <typename T,
  //           typename F,
  //           typename R = std::invoke_result_t<std::decay_t<F>, T, T, T>>
  // [[nodiscard]] multi_future<R> submit_blocks_with_tid(
  //     const T first_index,
  //     const T index_after_last,
  //     F &&block,
  //     const size_t num_blocks = 0) {
  //   multi_future<R> future_collection;

  //   if (index_after_last > first_index) {
  //     size_t M = num_blocks ? num_blocks : workers.size();
  //     T block_size = (index_after_last - first_index + M - 1) / M;  // Round
  //     up

  //     for (size_t i = 0; i < M; ++i) {
  //       T start = first_index + i * block_size;
  //       T end = std::min(start + block_size, index_after_last);

  //       if (start >= index_after_last) break;

  //       // Submit each block as a separate task and add the future to
  //       // multi_future
  //       future_collection.add(
  //           submit_task([block = std::forward<F>(block), start, end, i] {
  //             return block(start, end, i);
  //           }));
  //     }
  //   }
  //   return future_collection;
  // }

 private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queueMutex;
  std::condition_variable condition;
  bool stopFlag;
};

};  // namespace core