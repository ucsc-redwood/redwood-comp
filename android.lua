local ANDROID_CONFIG = {
    -- , "9b034f1b"
    ignored_devices = {"ZY22FLDDK7"},
    remote_base_path = "/data/local/tmp"  -- Base directory for all executables
}

-- ----------------------------------------------------------------------------
-- Common helper functions
-- ----------------------------------------------------------------------------

-- function parse_device_list()
--     -- Get connected devices
--     local devices_output = try { function()
--         return os.iorun("adb devices")
--     end}

--     if not devices_output then
--         raise("Failed to get device list from adb")
--     end

--     -- Parse device list
--     local devices = {}
--     for line in devices_output:gmatch("[^\r\n]+") do
--         if line:find("%s*device$") then
--             local device_id = line:match("(%S+)%s+device")
--             if device_id and not table.contains(ANDROID_CONFIG.ignored_devices, device_id) then
--                 table.insert(devices, device_id)
--             end
--         end
--     end

--     if #devices == 0 then
--         raise("No connected devices found!")
--     end

--     return devices
-- end

-- ----------------------------------------------------------------------------
-- Android deployment helper function
-- ----------------------------------------------------------------------------

function run_on_android(target)
    local exec_path = target:targetfile()
    local target_name = target:name()
    local remote_path = ANDROID_CONFIG.remote_base_path  -- .. "/" .. target_name
  
    if not os.isfile(exec_path) then
        raise("Executable not found at: " .. exec_path)
    end

    -- Get connected devices
    local devices_output = try { function()
        return os.iorun("adb devices")
    end}

    if not devices_output then
        raise("Failed to get device list from adb")
    end

    -- Parse device list
    local devices = {}
    for line in devices_output:gmatch("[^\r\n]+") do
        if line:find("%s*device$") then
            local device_id = line:match("(%S+)%s+device")
            if device_id and not table.contains(ANDROID_CONFIG.ignored_devices, device_id) then
                table.insert(devices, device_id)
            end
        end
    end

    if #devices == 0 then
        raise("No connected devices found!")
    end

    -- Run on each device
    import("core.base.option")
    local args = option.get("arguments") or {}

    for i, device_id in ipairs(devices) do
        print(string.format("[%d/%d] Running %s on device: %s", i, #devices, target_name, device_id))

        -- Deploy and execute
        local adb_commands = {
            {"-s", device_id, "push", exec_path, remote_path .. "/" .. target_name},
            {"-s", device_id, "shell", "chmod", "+x", remote_path .. "/" .. target_name},
        }

        -- Execute commands

        for _, cmd in ipairs(adb_commands) do
            if os.execv("adb", cmd) ~= 0 then
                print(string.format("Warning: Failed to execute adb command on device %s", device_id))
            end
        end

        -- Run the binary with arguments
        local run_command = {"-s", device_id, "shell", remote_path .. "/" .. target_name}

        table.join2(run_command, args, {"--device=" .. device_id})
        if os.execv("adb", run_command) ~= 0 then
            print(string.format("Warning: Failed to run %s on device %s", target_name, device_id))
        end

        print()
    end
end


-- ----------------------------------------------------------------------------
-- Push All resources to the device
-- ----------------------------------------------------------------------------

task("push-all-resources")
    set_menu {
        usage = "$project push-all-resources",
        description = "Push All resources to the device."
    }
    on_run(function ()
        local remote_path = ANDROID_CONFIG.remote_base_path 

        -- Get connected devices
        local devices_output = try { function()
            return os.iorun("adb devices")
        end}

        if not devices_output then
            raise("Failed to get device list from adb")
        end

        -- Parse device list
        local devices = {}
        for line in devices_output:gmatch("[^\r\n]+") do
            if line:find("%s*device$") then
                local device_id = line:match("(%S+)%s+device")
                if device_id and not table.contains(ANDROID_CONFIG.ignored_devices, device_id) then
                    table.insert(devices, device_id)
                end
            end
        end

        if #devices == 0 then
            raise("No connected devices found!")
        end

        -- For each device

        for i, device_id in ipairs(devices) do
            print(string.format("[%d/%d] Pushing all resources to device: %s", i, #devices, device_id))

            -- Pack and copy all resources to the device's remote path
            local data_dir = "./resources/"
            os.execv("zip", {"-r", data_dir .. "resources.zip", data_dir})
            os.execv("adb", {"-s", device_id, "push", data_dir .. "resources.zip", remote_path})
            os.execv("adb", {"-s", device_id, "shell", "unzip -o", remote_path .. "/resources.zip", "-d", remote_path})
            
        end

    end)
task_end()



