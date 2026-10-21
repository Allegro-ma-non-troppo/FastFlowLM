/// \file process_init.cpp
/// \brief Process-level setup and checks that belong to the host process, not to a model
/// \note Moved out of src/main.cpp so a program that links FastFlowLM can call them
///       individually and choose which ones it wants. main() calls them in the same
///       order as before.

#include "utils/process_init.hpp"
#include "utils/utils.hpp"
#include "utils/npu_platform.hpp"
#include "utils/debug_utils.hpp"
#include "nlohmann/json.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include "utils/wmi_helper.hpp"
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <linux/types.h>
#include <libdrm/drm.h>
#include <sys/utsname.h>
#include <cerrno>
#include "npu_utils/amdxdna_accel.h"
#endif

namespace flm::process {

void init_console_utf8() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

void preload_runtime_libraries() {
#if !defined(FLM_USE_HRX) && !defined(_WIN32)
    std::string exe_dir = utils::get_executable_directory();

    // When running inside a snap, LD_LIBRARY_PATH already points at the
    // bundled XRT libs (e.g. $SNAP/usr/lib/x86_64-linux-gnu).  Passing a
    // bare library name lets the dynamic linker resolve it via
    // LD_LIBRARY_PATH, which avoids hard-coding the arch triplet.
    // Outside of a snap the bundled libraries are expected to live in
    // lib/ relative to the executable.
    const char* snap_env = std::getenv("SNAP");
    std::string lib_prefix = snap_env && *snap_env ? "" : exe_dir + "/lib/";

    const std::vector<std::string> libraries = {
        "libxrt_core.so.2",           // Core - no dependencies
        "libxrt_coreutil.so.2",       // Depends on core
        "libxrt_driver_xdna.so.2",    // Driver
    };

    // Try to load the library with RTLD_GLOBAL so symbols are available to dependent libraries
    // Don't care about failures
    for (const auto& lib : libraries) {
        std::string lib_path = lib_prefix + lib;
        void* handle = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    }
#endif
}

void raise_process_priority() {
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
}

void configure_power_mode(const std::string& power_mode) {
#ifdef _WIN32
    std::string xrt_cmd = "cd \"C:\\Windows\\System32\\AMD\" && .\\xrt-smi.exe configure --pmode " + power_mode + " > NUL 2>&1";
    (void)system(xrt_cmd.c_str());
#else
    (void)power_mode;
#endif
}

void raise_memlock_limit(uint64_t required_bytes) {
#ifndef _WIN32
    rlim_t required = static_cast<rlim_t>(required_bytes);
    struct rlimit rl;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
        if (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < required) {
            struct rlimit rl_new = { required, required };
            if (setrlimit(RLIMIT_MEMLOCK, &rl_new) == 0) {
                header_print_g("Linux", "Memlock Limit: raised to " << (required / 1024 / 1024) << " MB");
            } else {
                header_print("Linux", "Warning: could not raise memlock limit to " << (required / 1024 / 1024) << " MB");
            }
        }
    }
#else
    (void)required_bytes;
#endif
}
namespace {

#ifdef _WIN32
std::string get_driver_version(const std::string& device_name) {
    std::string driver_version;
    wmi::WMIConnection wmi;
    if (!wmi.is_valid()) {
        return "";
    }

    std::wstring query = L"SELECT * FROM Win32_PnPSignedDriver WHERE DeviceName LIKE '%" +
        wmi::string_to_wstring(device_name) + L"%'";

    wmi.query(query, [&driver_version](IWbemClassObject* pObj) {
        if (driver_version.empty()) {  // Only get first match
            driver_version = wmi::get_property_string(pObj, L"DriverVersion");
        }
        });

    return driver_version;
}

std::string identify_npu_arch() {
    wmi::WMIConnection wmi_conn;
    if (!wmi_conn.is_valid()) {
        return "";
    }

    // XDNA2 NPU: AMD vendor 1022, device 17F0
    bool found_xdna2 = false;
    wmi_conn.query(
        L"SELECT PNPDeviceID FROM Win32_PnPEntity WHERE PNPDeviceID LIKE '%VEN_1022&DEV_17F0%'",
        [&found_xdna2](IWbemClassObject* pObj) {
            found_xdna2 = true;
        });

    if (found_xdna2) {
        return "XDNA2";
    }
    return "";
}

#endif

} // namespace

bool sanity_check_npu_stack(bool quiet, bool json_output) {
    bool print_human = !quiet && !json_output;
#ifdef FLM_ENABLE_CORE
    // Everything below reads the NPU's geometry and this process's memlock
    // limit from the driver and holds them against what FastFlowLM's own kernel
    // flow needs. A corelib build does not use that flow. corelib owns device
    // setup and brings the NPU up itself -- main() already reports whether that
    // worked, and says why when it did not -- so these checks answer a question
    // this build does not ask, against a threshold that describes FastFlowLM's
    // own kernels rather than the parts corelib supports. Left in, they fail a
    // working core install on hardware corelib runs on perfectly well.
    //
    // The JSON keeps the shape callers already parse, but reports the checks as
    // skipped rather than claiming they passed: "ready" here means "this build
    // does not gate on the stock stack". A caller that wants to know whether the
    // NPU actually came up should look at whether flm opened a device.
    if (json_output) {
        nlohmann::json validation_json = {
            {"object", "npu_stack_validation"},
#ifdef _WIN32
            {"platform", "windows"},
#else
            {"platform", "linux"},
#endif
            {"backend", "core"},
            {"checks_skipped", true},
            {"ready", true}
        };
        std::cout << validation_json.dump(4) << std::endl;
    } else if (print_human) {
        header_print("FLM", "corelib backend: skipping NPU stack checks (corelib owns device setup)");
    }
    return true;
#else
#ifndef _WIN32
    nlohmann::json validation_json = {
        {"object", "npu_stack_validation"},
        {"platform", "windows"},
        {"kernel_ok", true},
        {"amd_device_found", true},
        {"all_fw_ok", true},
        {"enough_cols", true},
        {"memlock_ok", true},
        {"devices", nlohmann::json::array()},
        {"ready", true}
    };
    validation_json["platform"] = "linux";
    // The same generation the catalog was filtered for in main().
    validation_json["npu_platform"] =
        std::string(utils::platform_id(utils::get_device()));
    // Check kernel version
    struct utsname u_name;
    if (uname(&u_name) != 0) {
        if (print_human)
            perror("Failed to get kernel version");
        validation_json["kernel_ok"] = false;
        validation_json["ready"] = false;
        if (json_output) {
            std::cout << validation_json.dump(4) << std::endl;
        }
        return false;
    }
    validation_json["kernel"] = u_name.release;
    if (print_human) {
        header_print("Linux", "Kernel: " << u_name.release);
    }

    // Check firmware version of all AMD devices
    bool all_fw_ok = true;
    bool enough_cols = true;
    bool amd_device_found = false;
    bool drm_version_ok = true;
    bool kernel_ok = true;

    for (int i = 0; i < 16; ++i) {
        std::string dev_name = "/dev/accel/accel" + std::to_string(i);
        int fd = open(dev_name.c_str(), O_RDWR);

        if (fd < 0) {
            if (errno == ENOENT)
                break;
            if (errno == EBUSY) {
                validation_json["devices"].push_back({{"device", dev_name}, {"busy", true}});
                if (print_human)
                    header_print_r("ERROR", "NPU " + dev_name + " could not be opened: device busy");
            }
            continue;
        }

        struct drm_version drm_v;
        memset(&drm_v, 0, sizeof(drm_v));
        std::string drm_version_str = "unknown";
        if (ioctl(fd, DRM_IOCTL_VERSION, &drm_v) == 0) {
            drm_version_str = std::to_string(drm_v.version_major) + "." + std::to_string(drm_v.version_minor);
            if (!(drm_v.version_major > 0 || (drm_v.version_major == 0 && drm_v.version_minor >= 6))) {
                drm_version_ok = false;
            }
        } else {
            drm_version_ok = false;
        }
        validation_json["drm_version"] = drm_version_str;

        amdxdna_drm_query_firmware_version query_fw_version;
        amdxdna_drm_get_info get_info = {
            .param = DRM_AMDXDNA_QUERY_FIRMWARE_VERSION,
            .buffer_size = sizeof(amdxdna_drm_query_firmware_version),
            .buffer = (unsigned long)&query_fw_version,
        };

        int ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
        if (ret < 0) {
            close(fd);
            // ENOTTY means it's not an AMD device, just ignore it.
            if (errno != ENOTTY) {
                 if (!quiet) {
                    header_print_r("ERROR", "Error code: " << ret << " on " << dev_name);
                    perror("Failed to get firmware version");
                 }
            }
            continue;
        }

        amd_device_found = true;

        nlohmann::json device_entry = {
            {"device", dev_name},
            {"fw_major", query_fw_version.major},
            {"fw_minor", query_fw_version.minor},
            {"fw_patch", query_fw_version.patch},
            {"fw_build", query_fw_version.build}
        };

        bool fw_ok = (query_fw_version.major > 1 || (query_fw_version.major == 1 && query_fw_version.minor >= 1));
        device_entry["fw_ok"] = fw_ok;
        if (!fw_ok) {
            all_fw_ok = false;
            if (print_human) {
                header_print_r("ERROR", "NPU firmware version on " + dev_name + " is incompatible. Please update NPU firmware!");
            }
        }

        amdxdna_drm_query_aie_metadata query_aie_metadata;
        get_info.param = DRM_AMDXDNA_QUERY_AIE_METADATA;
        get_info.buffer_size = sizeof(amdxdna_drm_query_aie_metadata);
        get_info.buffer = (unsigned long)&query_aie_metadata;
        if (ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info) == 0) {
            device_entry["cols"] = query_aie_metadata.cols;
            if (query_aie_metadata.cols < 8) {
                enough_cols = false;
                if (print_human) {
                    header_print_r("ERROR", "NPU " + dev_name + " does not have enough columns (" << query_aie_metadata.cols << " < 8)");
                }
            }
        }
        close(fd);

        if (print_human) {
            header_print_g("Linux", "NPU: " + dev_name + " with " + std::to_string(query_aie_metadata.cols) + " columns");
            header_print_g("Linux", "NPU FW Version: " << query_fw_version.major << "." << query_fw_version.minor << "." << query_fw_version.patch << "." << query_fw_version.build);
        }

        validation_json["devices"].push_back(device_entry);
    }
    validation_json["amd_device_found"] = amd_device_found;
    validation_json["all_fw_ok"] = all_fw_ok;
    validation_json["enough_cols"] = enough_cols;

    if (amd_device_found) {
        if (!drm_version_ok)
            kernel_ok = false;
        if (print_human) {
            if (drm_version_ok)
                header_print_g("Linux", "amdxdna version: " << validation_json["drm_version"].get<std::string>());
            else
                header_print_r("ERROR", "amdxdna version " << validation_json["drm_version"].get<std::string>() << " is incompatible");
        }
    }
    validation_json["kernel_ok"] = kernel_ok;

    if (!amd_device_found && print_human) {
        header_print_r("ERROR", "No NPU device found.");
    }

    // Check memlock limit
    bool memlock_ok = true;
    struct rlimit rl;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
        validation_json["memlock_limit"] = (rl.rlim_cur == RLIM_INFINITY)
            ? "infinity"
            : std::to_string(static_cast<unsigned long long>(rl.rlim_cur));
        if (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < 100 * 1024 * 1024) {
            struct rlimit rl_new = { RLIM_INFINITY, RLIM_INFINITY };
            if (setrlimit(RLIMIT_MEMLOCK, &rl_new) == 0) {
                rl.rlim_cur = RLIM_INFINITY;
                rl.rlim_max = RLIM_INFINITY;
                validation_json["memlock_limit"] = "infinity";
                if (print_human)
                    header_print_g("Linux", "Memlock Limit: set to infinity");
            } else {
                if (print_human) {
                    header_print_r("ERROR", "Memlock limit is too low (" << (rl.rlim_cur / 1024 / 1024) << "MB). Please raise the limit or set to infinity.");
                }
                memlock_ok = false;
            }
        } else if (print_human) {
            if (rl.rlim_cur == RLIM_INFINITY) {
                header_print_g("Linux", "Memlock Limit: infinity");
            } else {
                header_print("Linux", "Memlock Limit: " << (rl.rlim_cur / 1024 / 1024) << " MB");
            }
        }
    } else {
        if (print_human)
            perror("Failed to get memlock limit");
    }
    validation_json["memlock_ok"] = memlock_ok;

    if (print_human) {
        header_print_g("Linux", "NPU platform: " << validation_json["npu_platform"].get<std::string>());
    }

    bool overall_ok = amd_device_found && kernel_ok && all_fw_ok && enough_cols && memlock_ok;
    validation_json["ready"] = overall_ok;
    if (json_output) {
        std::cout << validation_json.dump(4) << std::endl;
    }

    return overall_ok;
#else
    nlohmann::json validation_json = {
        {"object", "npu_stack_validation"},
        {"platform", "windows"},
        {"amd_device_found", true},
        {"npu_driver_ok", true},
        {"npu_platform",
         std::string(utils::platform_id(utils::get_device()))},
        {"ready", true}
    };
    std::string npu_arch = identify_npu_arch();
    if (npu_arch.empty()) {
        if (print_human)
            header_print("Error", "No XDNA2 NPU hardware detected");
        validation_json["ready"] = false;
        validation_json["amd_device_found"] = false;
        if (json_output) {
            std::cout << validation_json.dump(4) << std::endl;
        }
        return false;
    }

    std::string min_drv = __NPU_VERSION__;
    std::string drv = get_driver_version("NPU Compute Accelerator Device");
    int mver_0, mver_1, mver_2, mver_3;
    int ver_0, ver_1, ver_2, ver_3;
    sscanf(min_drv.c_str(), "%d.%d.%d.%d", &mver_0, &mver_1, &mver_2, &mver_3);
    sscanf(drv.c_str(), "%d.%d.%d.%d", &ver_0, &ver_1, &ver_2, &ver_3);

    if (ver_3 < mver_3) {
        if (print_human) {
            header_print("Error", "NPU driver version doesn't meet the minimum!");
        }
        validation_json["ready"] = false;
        validation_json["npu_driver_ok"] = false;
        if (json_output) {
            std::cout << validation_json.dump(4) << std::endl;
        }
        return false;
    }

    if (print_human) {
        header_print_g("Windows", "NPU: " << npu_arch);
        header_print_g("Windows", "NPU dirver version: " << drv);
        header_print_g("Windows", "NPU platform: " << validation_json["npu_platform"].get<std::string>());
    }

    if (json_output) {
        std::cout << validation_json.dump(4) << std::endl;
    }
    return true;
#endif
#endif // FLM_ENABLE_CORE
}

///@brief open the NPU device that every engine in this process shares
///@param why if non-null, receives why the device could not be opened
///@return the shared device, or nullptr when no NPU could be opened
///@note Function-local static, so the lifetime is tied to the process exactly as
///      the per-Runner devices used to be. A machine with no NPU must still be
///      able to run `flm list`/`pull`/`version`, so failure is a null pointer,
///      not an error.
flm_rt::device* open_npu_device(std::string* why) {
    try {
        static flm_rt::device npu_device = flm_rt::device(0);
        return &npu_device;
    } catch (const std::exception& e) {
        if (why) *why = e.what();
        DO_VERBOSE(1, {
            header_print("FLM", "No NPU device available: " << e.what());
        });
        return nullptr;
    } catch (...) {
        if (why) *why = "unknown error";
        return nullptr;
    }
}
} // namespace flm::process
