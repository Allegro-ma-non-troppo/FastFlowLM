/// \file test_backend_package.cpp
/// \brief Native package preflight uses files, not the catalog's backend label
#include "AutoModel/flm_backend.hpp"
#include "../phi4_core/test_support.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>

namespace {
using flm::backend::PackageValidationStage;
using flm::backend::validate_flm_package;

void TestNativePackagePreflight() {
    const auto root = std::filesystem::temp_directory_path() /
        ("flm-native-package-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ec; std::filesystem::remove_all(path, ec); }
    } cleanup{root};
    std::filesystem::create_directories(root);
    const nlohmann::json oga = {{"backend", "oga-hipep"}, {"files", {"genai_config.json", "text.onnx"}}};
    RequireContains(RequireThrows([&] {
        validate_flm_package(oga, root.string(), PackageValidationStage::Manifest);
    }), "model.q4nx");
    validate_flm_package({{"files", {"config.json"}}}, root.string(), PackageValidationStage::Manifest);
    validate_flm_package({{"files", {"model.q4nx"}}}, root.string(), PackageValidationStage::Manifest);
    RequireContains(RequireThrows([&] {
        validate_flm_package(oga, root.string(), PackageValidationStage::Installed);
    }), "model.q4nx");
    for (const char* file : {"model.q4nx", "config.json", "tokenizer.json", "tokenizer_config.json"})
        std::ofstream(root / file) << "fixture";
    validate_flm_package(oga, root.string(), PackageValidationStage::Manifest);
    validate_flm_package(oga, root.string(), PackageValidationStage::Installed);
    const nlohmann::json vlm = {{"files", {"model.q4nx", "vision_weight.q4nx"}}};
    RequireContains(RequireThrows([&] {
        validate_flm_package(vlm, root.string(), PackageValidationStage::Installed);
    }), "vision_weight.q4nx");
}
}  // namespace

int main() {
    RunTest(TestNativePackagePreflight, "native package preflight");
    return 0;
}
