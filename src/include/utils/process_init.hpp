/// \file process_init.hpp
/// \brief Process-level setup and checks that belong to the host process, not to a model
/// \note These used to be written inline in src/main.cpp. A program that links FastFlowLM
///       into its own process calls the ones it wants and leaves the rest: the console
///       code page, the process priority and the NPU power mode are the host's to decide.
#pragma once

#include <cstdint>
#include <string>

#include "device_runtime.hpp"

namespace flm::process {

///@brief switch the Windows console to UTF-8 input and output; does nothing elsewhere
void init_console_utf8();

///@brief Linux, XRT backend: dlopen the XRT libraries bundled next to the executable
///@note Failures are ignored. Does nothing on Windows or in an HRX build.
void preload_runtime_libraries();

///@brief open the NPU device that every engine in this process shares
///@param why if non-null, receives why the device could not be opened
///@return the shared device, or nullptr when no NPU could be opened
///@note Function-local static: a process can open only one NPU device, so a second
///      xrt::device for the same NPU must never be created. Failure is a null pointer
///      so a machine with no NPU can still run the commands that need none.
flm_rt::device* open_npu_device(std::string* why);

///@brief Windows: raise the process to HIGH_PRIORITY_CLASS; does nothing elsewhere
void raise_process_priority();

///@brief Windows: run `xrt-smi configure --pmode <power_mode>`; does nothing elsewhere
///@param power_mode one of default, powersaver, balanced, performance, turbo
void configure_power_mode(const std::string& power_mode);

///@brief Linux: raise RLIMIT_MEMLOCK to at least required_bytes when it is lower
///@note Prints the outcome like flm does; does nothing on Windows.
void raise_memlock_limit(uint64_t required_bytes);

///@brief check the NPU stack (driver, firmware, kernel, memlock) as `flm validate` does
///@param quiet suppress the human-readable report
///@param json_output print the npu_stack_validation JSON object on stdout
///@return true when the stack is ready
bool sanity_check_npu_stack(bool quiet, bool json_output = false);

} // namespace flm::process
