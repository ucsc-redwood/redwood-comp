import re

def parse_benchmark_line(line):
    parts = line.strip().split()
    if len(parts) < 4:
        return None

    # Extract time value from the line
    rev = parts[::-1]
    try:
        time_val = float(rev[4])  # The Time value
    except ValueError:
        return None

    # Parse the benchmark name
    name_tokens = parts[:-5]
    benchmark_name = " ".join(name_tokens)
    name_parts = benchmark_name.split('/')

    run_mode = name_parts[0]

    # Remove iterations:xxx if present at the end
    if name_parts[-1].startswith("iterations:"):
        name_parts = name_parts[:-1]

    stage_name = None
    core_type = None
    threads = None

    if run_mode.startswith("CPU"):
        # CPU pattern
        if len(name_parts) == 2:
            stage_name = name_parts[1]
        elif len(name_parts) == 3:
            stage_name = name_parts[1]
            core_type = name_parts[2]
        elif len(name_parts) == 4:
            stage_name = name_parts[1]
            core_type = name_parts[2]
            threads = int(name_parts[3])
        else:
            stage_name = name_parts[1]
            # Try parsing threads from last
            try:
                threads = int(name_parts[-1])
                core_type = "/".join(name_parts[2:-1]) if len(name_parts[2:-1]) > 0 else None
            except ValueError:
                core_type = "/".join(name_parts[2:])
    else:
        # GPU pattern: iGPU_CUDA or iGPU_Vulkan
        # No CPU cores/threads. Set core_type = "GPU"
        if len(name_parts) >= 2:
            stage_name = name_parts[1]
        core_type = "GPU"

    data = {
        "run_mode": run_mode,
        "stage": stage_name,
        "core_type": core_type,
        "threads": threads,
        "time": time_val
    }
    return data

def parse_application_and_backend_from_cmd(cmd_line):
    parts = cmd_line.strip().split()
    binary = None
    device = None
    for i, p in enumerate(parts):
        if p.startswith("bm-"):
            binary = p
        if p == "-d" and i+1 < len(parts):
            device = parts[i+1]

    if not binary:
        return None, None, None

    bin_parts = binary.split('-')
    if len(bin_parts) < 3:
        # Maybe bm-tree or something else
        if len(bin_parts) == 2:
            application = bin_parts[1]
            backend = None
        else:
            application = None
            backend = None
    else:
        backend = bin_parts[-1]
        application = "-".join(bin_parts[1:-1])

    # If device is "jetson", rename it to "Jetson Orin Nano"
    if device == "jetson":
        device = "Jetson Orin Nano"

    return application, backend, device

def parse_file(filepath):
    data_structure = {}
    current_application = None
    current_backend = None
    current_device = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Detect a command line invocation
            if line.startswith("xmake r "):
                # Parse application, backend, device
                app, be, dev = parse_application_and_backend_from_cmd(line)
                if dev is None:
                    dev = "unknown_device"

                current_application = app
                current_backend = be
                current_device = dev

                if current_device not in data_structure:
                    data_structure[current_device] = {}
                if current_application not in data_structure[current_device]:
                    data_structure[current_device][current_application] = {}
                if current_backend not in data_structure[current_device][current_application]:
                    data_structure[current_device][current_application][current_backend] = {}

            # Check if we have a line that identifies the device in Android scenario
            # e.g. "[1/2] Running bm-cifar-dense-cpu on device: 3A021JEHN02756"
            elif "on device:" in line:
                # Extract the device ID from this line
                # Format: ... on device: XXX
                match = re.search(r"on device:\s+(\S+)", line)
                if match:
                    found_device = match.group(1)
                    # This is an android device ID, use it
                    current_device = found_device
                    if current_device not in data_structure:
                        data_structure[current_device] = {}
                    if current_application and current_backend:
                        if current_application not in data_structure[current_device]:
                            data_structure[current_device][current_application] = {}
                        if current_backend not in data_structure[current_device][current_application]:
                            data_structure[current_device][current_application][current_backend] = {}

            # Parse benchmark lines
            elif (line.startswith("CPU_Pinned") or
                  line.startswith("CPU_Unpinned") or
                  line.startswith("iGPU_CUDA") or
                  line.startswith("iGPU_Vulkan")):
                parsed = parse_benchmark_line(line)
                if parsed and current_application and current_backend and current_device:
                    run_mode = parsed["run_mode"]
                    stage = parsed["stage"]
                    core_type = parsed["core_type"]
                    threads = parsed["threads"]
                    time_val = parsed["time"]

                    dev_dict = data_structure[current_device][current_application][current_backend]
                    if run_mode not in dev_dict:
                        dev_dict[run_mode] = {}
                    mode_dict = dev_dict[run_mode]

                    if stage not in mode_dict:
                        mode_dict[stage] = {}

                    if run_mode.startswith("CPU"):
                        ckey = (core_type, threads)
                    else:
                        # GPU run, we set core_type = "GPU"
                        ckey = (core_type, None)  # keep a consistent tuple shape

                    mode_dict[stage][ckey] = time_val

    return data_structure


if __name__ == "__main__":
    # Example usage:
    # For Jetson:
    jetson_results = parse_file('jetson.txt')
    # For Android:
    android_results = parse_file('androids.txt')

    import pprint
    print("Jetson results:")
    pprint.pprint(jetson_results)
    print("\nAndroid results:")
    pprint.pprint(android_results)
