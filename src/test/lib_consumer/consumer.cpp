/// \file consumer.cpp
/// \brief A program that embeds FastFlowLM as a library, built without the flm CLI.
/// \note Built by CMakeLists.txt through add_subdirectory(<FastFlowLM>/src) with
///       FLM_BUILD_CLI=OFF, and linked against flm_lib. It calls two kinds of code:
///       functions compiled into flm_lib (npu_platform.cpp, utils.cpp) and the
///       header-only model_list over the shipped catalog. It never touches the
///       NPU: get_device() is the stand-in that consults nothing, and no model is
///       loaded.
#include "model_list.hpp"
#include "utils/npu_platform.hpp"
#include "utils/utils.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (condition) {
        std::cout << "PASS " << what << '\n';
    } else {
        std::cout << "FAIL " << what << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    // Compiled into flm_lib (src/common/npu_platform.cpp).
    check(utils::parse_platform("aie2p") == utils::npu_platform::aie2p,
          "flm_lib: parse_platform(\"aie2p\") returns aie2p");
    check(utils::platform_id(utils::npu_platform::dev04) == "dev04",
          "flm_lib: platform_id(dev04) is \"dev04\"");
    check(utils::get_device() == utils::default_npu_platform(),
          "flm_lib: get_device() is the stand-in default (no probe)");

    // Compiled into flm_lib (src/common/utils.cpp).
    check(!utils::get_executable_directory().empty(),
          "flm_lib: get_executable_directory() is non-empty");

    // Header-only model_list over the catalog shipped beside the sources.
    std::string list_path = FLM_CONSUMER_MODEL_LIST_PATH;
    std::string exe_dir = ".";
    model_list models(list_path, exe_dir, "aie2p", {"flm"});
    check(models.is_model_supported("phi4-mini-it:4b"),
          "model_list: the FastFlowLM phi4 entry is offered on aie2p with the flm kernels");
    check(!models.is_model_supported("phi4-mini-it-core:4b"),
          "model_list: the corelib phi4 entry is pruned when core is not linked");

    if (failures != 0) {
        std::cout << failures << " check(s) FAILED\n";
        return EXIT_FAILURE;
    }
    std::cout << "flm_lib consumer OK\n";
    return EXIT_SUCCESS;
}
