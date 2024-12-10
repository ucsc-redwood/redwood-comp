#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

struct CoreInfo {
  int processor_id;
  int frequency_max;
};

[[nodiscard]] std::string get_cpu_policy_max_freq(const int cpu_id) {
  std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu_id) +
                     "/cpufreq/scaling_max_freq";
  std::ifstream freq_file(path);
  std::string freq;
  if (freq_file >> freq) {
    return freq;
  }
  return "0";  // Return 0 if can't read frequency
}

void print_core_info() {
  // First get number of CPU cores
  int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
  std::vector<CoreInfo> cores;

  std::cout << "\nDetected " << num_cpus << " CPU cores\n" << std::endl;

  // Collect information for each core
  for (int i = 0; i < num_cpus; i++) {
    CoreInfo core;
    core.processor_id = i;
    std::string max_freq = get_cpu_policy_max_freq(i);
    core.frequency_max = std::stoi(max_freq);
    cores.push_back(core);

    // Print individual core info
    std::cout << "Core " << i
              << ": max frequency = " << core.frequency_max / 1000 << " MHz"
              << std::endl;
  }

  // Group cores by frequency
  std::map<int, std::vector<int>> freq_groups;
  for (const auto& core : cores) {
    freq_groups[core.frequency_max].push_back(core.processor_id);
  }

  std::cout << "\nCore Type Analysis:" << std::endl;
  std::cout << "-----------------" << std::endl;

  if (freq_groups.size() == 1) {
    std::cout
        << "Homogeneous architecture detected (all cores are the same type)"
        << std::endl;
    std::cout << "Cores: {";
    for (int i = 0; i < num_cpus; i++) {
      std::cout << i;
      if (i < num_cpus - 1) std::cout << ", ";
    }
    std::cout << "}" << std::endl;
  } else {
    std::cout << "Heterogeneous architecture detected" << std::endl;

    // Print in reverse order (highest frequency first)
    for (auto it = freq_groups.rbegin(); it != freq_groups.rend(); ++it) {
      std::string type;
      if (it == freq_groups.rbegin()) {
        type = "big";
      } else if (std::next(it) == freq_groups.rend()) {
        type = "small";
      } else {
        type = "medium";
      }

      std::cout << type << " cores (" << it->first / 1000 << " MHz): {";
      for (size_t i = 0; i < it->second.size(); i++) {
        std::cout << it->second[i];
        if (i < it->second.size() - 1) std::cout << ", ";
      }
      std::cout << "}" << std::endl;
    }
  }
}

int main() {
  std::cout << "CPU Core Information:" << std::endl;
  std::cout << "====================" << std::endl;

  print_core_info();

  return 0;
}
