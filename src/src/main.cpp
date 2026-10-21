/// \file main.cpp
/// \brief Main entry point for the FLM application
/// \author FastFlowLM Team
/// \date 2025-08-05
/// \version 0.9.24
/// \note This is a source file for the main entry point
#pragma once
#include "runner.hpp"
#include "server.hpp"
#include "model_list.hpp"
#include "AutoModel/model_backend.hpp"
#include "model_downloader.hpp"
#include "update.hpp"
#include "utils/utils.hpp"
#include "utils/npu_platform.hpp"
#include "utils/process_init.hpp"
#include "program_args.hpp"
#include "minja/chat-template.hpp"
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <filesystem>
#include <cstdlib>
#include <optional>
#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#include <shellapi.h>
#include <io.h>
#include "utils/wmi_helper.hpp"
#endif
#include "utils/vm_args.hpp"
#include <boost/program_options.hpp>
#include "benchmarking.hpp"
#ifdef FLM_ENABLE_CORE
#include "core/corelib_runtime.hpp"
#include "core/corelib_device.hpp"
#endif

#ifndef _WIN32
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

#include "AutoModel/automodel.hpp"

// Global variables
///@brief should_exit is used to control the server thread
std::atomic<bool> should_exit(false);
std::mutex exit_mutex;
std::condition_variable exit_cv;


#ifdef _WIN32
///@brief get_unicode_command_line_args gets Unicode command line arguments
///@param argc_out reference to store argument count
///@return vector of UTF-8 encoded argument strings
std::vector<std::string> get_unicode_command_line_args(int& argc_out) {
    std::vector<std::string> args;
    
    // Get the Unicode command line
    LPWSTR* szArglist;
    int nArgs;
    
    szArglist = CommandLineToArgvW(GetCommandLineW(), &nArgs);
    if (szArglist == nullptr) {
        argc_out = 0;
        return args;
    }
    
    argc_out = nArgs;
    
    // Convert each argument from wide string to UTF-8
    for (int i = 0; i < nArgs; i++) {
        std::wstring warg(szArglist[i]);
        
        // Convert to UTF-8
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, warg.c_str(), (int)warg.size(), nullptr, 0, nullptr, nullptr);
        if (size_needed > 0) {
            std::string utf8_arg(size_needed, 0);
            WideCharToMultiByte(CP_UTF8, 0, warg.c_str(), (int)warg.size(), &utf8_arg[0], size_needed, nullptr, nullptr);
            args.push_back(utf8_arg);
        } else {
            args.push_back(""); // fallback for conversion error
        }
    }
    
    // Free memory allocated by CommandLineToArgvW
    LocalFree(szArglist);
    
    return args;
}
#endif

///@brief get_user_documents_directory gets the user's Documents directory
///@return the user's Documents directory path


///@brief ensure_models_directory creates the models directory if it doesn't exist
///@param exe_dir the executable directory
void ensure_models_directory(const std::string& exe_dir) {
    // Use Documents/flm/models directory on Windows or ~/.config/flm on Linux for models instead of executable directory
    std::string models_dir = utils::get_models_directory();
    if (!std::filesystem::exists(models_dir)) {
        std::filesystem::create_directories(models_dir);
    }
}

std::atomic<bool> running(true);
std::condition_variable cv;
std::mutex mtx;
void signal_handler(int signal) {
    if (signal == SIGINT) {
        running = false;
        cv.notify_one();
    }
}

///@brief create_lm_server is used to create the ollama server
///@param models the model list
///@param default_tag the default tag
///@param port the port to listen on, default is 52625, same with the ollama server
///@param npu_device the NPU device owned by main, shared with the handler
///@return the server
std::unique_ptr<WebServer> create_lm_server(model_list& models, ModelDownloader& downloader, program_args_t& args, flm_rt::device* npu_device);


#ifdef FLM_ENABLE_CORE
/// \brief brings corelib up for the process and tears it down on every exit path
/// \note main() has early `return 1`s and a catch-all, so the teardown has to be
///       a destructor rather than a trailing call. A failed shutdown must not
///       mask whatever the program was already reporting, hence the swallow.
struct CoreProcessGuard {
    ~CoreProcessGuard() {
        try {
            flm::corelib::CorelibRuntime::ShutdownProcess();
        } catch (const std::exception& e) {
            std::cerr << "Warning: corelib shutdown failed: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Warning: corelib shutdown failed" << std::endl;
        }
    }
};
#endif

///@brief main function
///@param argc the number of arguments
///@param argv the arguments
///@return the exit code
int main(int argc, char* argv[]) {
    flm::process::init_console_utf8();
    // XRT backend on Linux: preload the bundled XRT libraries from the executable directory.
    flm::process::preload_runtime_libraries();
    
    // Parse command line arguments using Boost Program Options
    program_args_t parsed_args;
    if (!arg_utils::parse_options(argc, argv, parsed_args)) {
        return 1;
    }

    if (parsed_args.command == "help") {
        return 0;
    }

    if (parsed_args.command == "version") {
        if (parsed_args.json_output) {
            std::cout << "{ \"version\": \"" << __FLM_VERSION__ << "\" }" << std::endl;
        } else {
            std::cout << "FLM v" << __FLM_VERSION__ << std::endl;
        }
        return 0;
    }

    if (parsed_args.command == "port") {
        if (parsed_args.json_output) {
            std::cout << "{ \"port\": " << utils::get_server_port(parsed_args.port) << " }" << std::endl;
        } else {
            std::cout << "Server Port: " << utils::get_server_port(parsed_args.port) << std::endl;
        }
        return 0;
    }

    // Get the command, model tag, and force flag
    std::string exe_dir = utils::get_executable_directory();
    std::string config_path;
    try {
        config_path = utils::find_model_list();
        // header_print("FLM", "Fetching models from: " + config_path);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    // Get the models directory from environment variable or default
    std::string models_dir = utils::get_models_directory();

    // One NPU runtime for the whole process, brought up before the catalog so
    // the catalog can be filtered to what this NPU generation can actually run.
    // Which runtime that is follows from the build, not from a decision here:
    // the flm backend opens its own device, core gets one from corelib.
#ifdef FLM_ENABLE_CORE
    // Declared before anything that can return so the destructor covers the
    // early exits and the catch-all below.
    CoreProcessGuard core_guard;
    // Declared out here so the early exits below still see a device, and so a
    // corelib that fails to come up leaves it null rather than undefined.
    flm_rt::device* npu_device = nullptr;
    // Kept so the reason survives into a release build. VERBOSE is a
    // compile-time macro, so every DO_VERBOSE below is compiled out and a user
    // is left with "NPU device instance is nullptr" -- raised much later, by
    // whichever model reaches for the device first -- and nothing to act on.
    std::string npu_open_error;
    // Set when corelib failed but a direct device was opened anyway: the core
    // backend is gone, the rest of the build is not.
    std::string corelib_note;

    try {
        const auto runtime =
            flm::corelib::CorelibRuntime::GetOrCreate(std::filesystem::path(exe_dir));
        // corelib opens the NPU for this process, so take its device rather
        // than opening a second one: a buffer object created against a
        // different xrt::device for the same NPU binds without error and then
        // never completes. GetOrCreate holds the runtime process-wide, so the
        // device stays valid until CoreProcessGuard tears it down at exit.
        npu_device = flm::corelib::SharedDevice(*runtime);
        if (npu_device == nullptr)
            npu_open_error =
                "corelib started but reports no NPU device on this machine";
    } catch (const std::exception& e) {
        // A box with no NPU must still run `flm list`/`pull`/`version`, so this
        // stays a null device rather than an error, as in the non-core path.
        npu_open_error = std::string("corelib unavailable: ") + e.what();
        header_print("FLM", "corelib unavailable: " << e.what());
    }
    if (npu_device == nullptr) {
        // corelib having no device is a reason for the core backend to be
        // unavailable, not for the process to have no NPU at all. Every other
        // family is served by the flm backend, which opens its own device and
        // never touches corelib -- so opening one directly keeps those models
        // working instead of failing the whole process over a backend they do
        // not use. On a box whose NPU has no creatable DEV04 hw_context this is
        // the difference between "core is unavailable" and "flm has no NPU".
        //
        // This does not bring back the two-device bug, where a buffer created
        // against a second xrt::device for the same NPU bound without error and
        // then never completed. That needed corelib to be live and holding a
        // device of its own; here it has none, so the process still ends up
        // with exactly one.
        corelib_note = npu_open_error;
        std::string raw_error;
        npu_device = flm::process::open_npu_device(&raw_error);
        if (npu_device != nullptr) {
            npu_open_error.clear();
        } else {
            npu_open_error += "; opening the NPU directly also failed: " + raw_error;
            corelib_note.clear();
        }
    }
#else
    // validate probes /dev/accel/* itself, and amdxdna refuses a second open of
    // the same device from one process: run it before this process takes the
    // device, or every probe gets EBUSY and the check reports "No NPU device found".
    if (parsed_args.command == "validate") {
        return flm::process::sanity_check_npu_stack(false, parsed_args.json_output) ? 0 : 1;
    }

    std::string npu_open_error;
    std::string corelib_note;
    flm_rt::device* npu_device = flm::process::open_npu_device(&npu_open_error);
#endif

    // Two things decide what this binary can run, and they are independent.
    // The generation is the machine's, answered by the device itself, and says
    // which artifacts run here at all; the backend ids are the kernel flows
    // that were linked in, and a tag names its own flow.
    const utils::npu_platform platform = utils::get_device();

    // The constructor throws when the catalog cannot be opened; report it the
    // way every other startup failure is reported rather than letting it escape.
    std::optional<model_list> availble_models_storage;
    try {
        availble_models_storage.emplace(
            config_path, models_dir, std::string(utils::platform_id(platform)),
            flm::backend::BackendRegistry::instance().backend_ids());
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    model_list& availble_models = *availble_models_storage;

    const bool print_status = !parsed_args.json_output && !parsed_args.sub_process_mode;
    const bool needs_npu =
        parsed_args.command == "run" || parsed_args.command == "serve" ||
        parsed_args.command == "bench" || parsed_args.command == "validate";

    // Say once, here, why there is no device. The commands below cannot run
    // without one, and the message they eventually produce names the symptom
    // rather than the cause: it is raised from whichever model first reaches
    // for the device, long after the runtime that failed to come up.
    if (print_status && needs_npu && !corelib_note.empty()) {
        header_print("FLM", "core backend unavailable ("
                                << corelib_note
                                << "); opened the NPU directly");
    }
    if (needs_npu && npu_device == nullptr) {
        header_print("ERROR", "No NPU device available"
                                  << (npu_open_error.empty()
                                          ? std::string()
                                          : ": " + npu_open_error));
    }

    // Extract parsed values
    bool got_power_mode = (parsed_args.power_mode != "performance"); // Check if user explicitly set power mode
    bool stable_stack = false;

    // Set process priority to high for better performance
    flm::process::raise_process_priority();

    // Check if the commands and args valid
    if (parsed_args.command == "validate") {
        stable_stack = flm::process::sanity_check_npu_stack(parsed_args.command != "validate", parsed_args.command == "validate" && parsed_args.json_output);
        return stable_stack ? 0 : 1;
    }

    if (parsed_args.command == "run" || parsed_args.command == "serve" || parsed_args.command == "pull" || parsed_args.command == "remove" || parsed_args.command == "check" || parsed_args.command == "bench") {
      if (parsed_args.model_tag != "model-faker" && (!availble_models.is_model_supported(parsed_args.model_tag))) {
            header_print("ERROR", "Model not found: " << parsed_args.model_tag << "; Please check with `flm list` and try again.");
            return 1;
        }
    }
  
    if (parsed_args.command == "serve" || parsed_args.command == "run" || parsed_args.command == "bench"){
        // Configure AMD XRT for the specified power mode
        if (parsed_args.power_mode == "default" || parsed_args.power_mode == "powersaver" || parsed_args.power_mode == "balanced" || 
            parsed_args.power_mode == "performance" || parsed_args.power_mode == "turbo") {
#ifdef _WIN32
            header_print("FLM", "Configuring NPU Power Mode to " + parsed_args.power_mode + (got_power_mode ? "" : " (flm default)"));
            flm::process::configure_power_mode(parsed_args.power_mode);
#endif
        }
        else{
            std::cout << "Invalid power mode: " << parsed_args.power_mode << std::endl;
            std::cout << "Valid power modes: default, powersaver, balanced, performance, turbo" << std::endl;
            return 1;
        }
    }

    // Handle special case for serve command - use default tag if none provided
    if (parsed_args.command == "serve" && parsed_args.model_tag.empty()) {
        parsed_args.model_tag = "model-faker"; // Use default tag
    }

#ifndef _WIN32
    // Raise memlock limit to accommodate the model being loaded
    if ((parsed_args.command == "run" || parsed_args.command == "serve" || parsed_args.command == "bench")) {
        rlim_t model_size = 0;
        rlim_t asr_size = 0;
        rlim_t embedding_size = 0;

        if (parsed_args.model_tag != "model-faker" && !parsed_args.model_tag.empty()) {
            auto [resolved_tag, model_info] = availble_models.get_model_info(parsed_args.model_tag);
            if (model_info.contains("size")) {
                model_size = model_info["size"].get<uint64_t>();
            }
        }
        if (parsed_args.asr) {
            asr_size = 1000000000;
            header_print_g("FLM", "ASR mode enabled: reserving additional 1GB of memory");
        }
        if (parsed_args.embed) {
            embedding_size = 300000000;
            header_print_g("FLM", "Embedding mode enabled: reserving additional 300MB of memory");
        }

        // Add 512MB overhead for working memory beyond the model itself
        rlim_t required = model_size + asr_size + embedding_size + (512ULL * 1024 * 1024);
        flm::process::raise_memlock_limit(required);
    }
#endif

    // code for all commands:

    if (parsed_args.preemption){
        header_print("FLM", "Allowing high priority tasks to preempt FLM!");
    }


    try {

        // Load the model list with the models directory as the base
        ModelDownloader downloader(availble_models);

        // Ensure models directory exists
        if (!std::filesystem::exists(models_dir)) {
            std::filesystem::create_directories(models_dir);
        }

        if (parsed_args.command == "bench") {
            benchmarking::BenchmarkResults_t results = benchmarking::run_benchmarks(parsed_args.model_tag, parsed_args.input_file_name, availble_models, parsed_args.iterations, npu_device, parsed_args.backend);
        }
        else if (parsed_args.command == "run") {
            check_and_notify_new_version();
            Runner runner(availble_models, downloader, parsed_args, npu_device);
            runner.run();

        } else if (parsed_args.command == "serve") {
            check_and_notify_new_version();
            // Create the server
            int port = utils::get_server_port(parsed_args.port);
            if (parsed_args.port == -1) { // User did not specify port
                header_print("FLM", "Using environment-specified port: " << port);
                parsed_args.port = port; // overwrite to ensure server uses correct port
            } else {
                header_print("FLM", "Using user-specified port: " << port);
            }
            auto server = create_lm_server(availble_models, downloader, parsed_args, npu_device);
            server->set_max_connections(parsed_args.max_socket_connections);           // Allow up to 10 concurrent connections
            server->set_io_threads(10);          // Allow up to 5 io threads
            server->set_npu_queue_length(parsed_args.max_npu_queue);           // Allow up to 10 concurrent queue
            server->set_request_timeout(std::chrono::seconds(600)); // 10 minute timeout for long requests
            // Start the server
            header_print("FLM", "Starting server on port " << port << "...");
            server->start();
            header_print("FLM", "Press Ctrl+C to stop.");
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [] { return !running.load(); });
            }
            // header_print("FLM", "Stopping server...");
            // server->stop();
        }
        else if (parsed_args.command == "pull") {
                bool success = downloader.pull_model(parsed_args.model_tag, parsed_args.modelscope, parsed_args.force_redownload);
                if (!success) {
                    header_print("ERROR", "Failed to pull model: " + parsed_args.model_tag);
                    return 1; 
                }
        }
        else if (parsed_args.command == "remove") {
            // Remove the model, this will be used to remove the model
            downloader.remove_model(parsed_args.model_tag);
        }
        else if (parsed_args.command == "check") {
            downloader.check_model(parsed_args.model_tag, parsed_args.modelscope);
        }
        else if (parsed_args.command == "list") {
            // List the models, this will be used to list the models
            if (!(parsed_args.list_filter == "installed" || parsed_args.list_filter == "not-installed" || parsed_args.list_filter == "all")) {
                header_print("Error", "Invalid filter: please use 'all', 'installed', or 'not-installed'");
                return 1;
            }
            if (parsed_args.json_output) {
                nlohmann::json output_json;
                output_json["models"] = nlohmann::json::array();
                nlohmann::json models = availble_models.get_all_models();
                for (const auto& model : models["models"]) {
                    // Fast path: `list` only needs presence + compatibility,
                    // so skip HF metadata fetch and per-file hash verification.
                    ModelDownloader::ModelStatus status = downloader.is_model_downloaded(model["name"].get<std::string>(), /*parsed_args.sub_process_mode*/true, /*fast_check=*/true);
                    bool is_present = (status == ModelDownloader::ModelStatus::Ready);
                    if ((parsed_args.list_filter == "installed") == is_present || parsed_args.list_filter == "all") {
                        nlohmann::json model_entry = model;
                        model_entry["installed"] = is_present;
                        output_json["models"].push_back(model_entry);
                    }
                }
                std::cout << output_json.dump(4) << std::endl;
            }
            else {
                std::cout << "Models:" << std::endl;
                nlohmann::json models = availble_models.get_all_models();
                bool any_model_listed = false;
                for (const auto& model : models["models"]) {
                    // Fast path: skip HF metadata fetch and per-file hash verification.
                    ModelDownloader::ModelStatus status = downloader.is_model_downloaded(model["name"].get<std::string>(), parsed_args.sub_process_mode, /*fast_check=*/true);
                    bool is_present = (status == ModelDownloader::ModelStatus::Ready);
                    if ((parsed_args.list_filter == "installed") == is_present || parsed_args.list_filter == "all") {
                        std::cout << "  - " << model["name"].get<std::string>();
                        if (!parsed_args.sub_process_mode) {
                            switch (status) {
                                case ModelDownloader::ModelStatus::Ready:        std::cout << " ✅"; break;
                                case ModelDownloader::ModelStatus::Missing:      std::cout << " ⏬"; break;
                                case ModelDownloader::ModelStatus::Outdated:     std::cout << " ⚠️"; break;
                                case ModelDownloader::ModelStatus::Incompatible: std::cout << " ⚠️"; break;
                            }
                        }
                        std::cout << std::endl;
                        any_model_listed = true;
                    }
                }
                if (!any_model_listed) {
                    std::cout << "  No models found for the specified filter." << std::endl;
                }
            }
        }
        else {
            // Invalid command, this will be used to show the invalid command
            std::cerr << "Invalid command: " << parsed_args.command << std::endl;
            std::cerr << "Use --help for usage information" << std::endl;
            return 1;
        }
        // Return 0 if the command is valid; corelib_guard shuts corelib down.
        return 0;
    } catch (const std::exception& e) {
        // If an error occurs, this will be used to show the error
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
