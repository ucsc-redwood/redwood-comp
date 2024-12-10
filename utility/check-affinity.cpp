#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

void pin_thread_to_core(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
    throw std::runtime_error("Failed to pin thread to core " +
                             std::to_string(core_id));
  }
}

int main() {
  auto n_threads = std::thread::hardware_concurrency();
  std::vector<bool> pinnable_cores(n_threads, false);

  // Reset affinity to all cores first
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (auto i = 0u; i < n_threads; i++) {
    CPU_SET(i, &cpuset);
  }
  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
    throw std::runtime_error("Failed to reset CPU affinity to all cores");
  }

  // Test pinning on each core
  for (auto i = 0u; i < n_threads; i++) {
    try {
      pin_thread_to_core(i);
      pinnable_cores[i] = true;
    } catch (const std::runtime_error&) {
      pinnable_cores[i] = false;
    }
  }

  // Print results in a formatted way
  std::cout << "\nCPU Affinity Test Results\n";
  std::cout << "========================\n\n";

  // Count available cores
  int available_cores = 0;
  for (bool pinnable : pinnable_cores) {
    if (pinnable) available_cores++;
  }

  std::cout << "Total logical cores: " << n_threads << "\n";
  std::cout << "Pinnable cores:      " << available_cores << "\n\n";

  std::cout << "Core Status:\n";
  std::cout << "-----------\n";

  const int CORES_PER_ROW = 8;
  for (auto i = 0u; i < n_threads; i++) {
    if (i % CORES_PER_ROW == 0) {
      std::cout << "\nCore " << std::setw(2) << i << "-" << std::setw(2)
                << std::min(i + CORES_PER_ROW - 1, n_threads - 1) << ": ";
    }

    if (pinnable_cores[i]) {
      std::cout << "\033[32m" << "✓ " << "\033[0m";  // Green checkmark
    } else {
      std::cout << "\033[31m" << "✗ " << "\033[0m";  // Red X
    }
  }
  std::cout << "\n\nLegend: \033[32m✓\033[0m = Pinnable  \033[31m✗\033[0m = "
               "Not Pinnable\n\n";

  // Print pinnable core IDs in vector format
  std::cout << "Pinnable core IDs: {";
  bool first = true;
  for (auto i = 0u; i < n_threads; i++) {
    if (pinnable_cores[i]) {
      if (!first) {
        std::cout << ", ";
      }
      std::cout << i;
      first = false;
    }
  }
  std::cout << "}\n\n";

  return EXIT_SUCCESS;
}
