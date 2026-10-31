/// \file flm_backend.cpp
/// \brief Read-only checks for the native FLM model package
#include "AutoModel/flm_backend.hpp"
#include "AutoModel/backend_package.hpp"

namespace flm::backend {

void validate_flm_package(const nlohmann::json& info, const std::string& path,
                          PackageValidationStage stage) {
    require_package_file(info, path, "model.q4nx", stage);
    if (stage == PackageValidationStage::Manifest) return;
    for (const char* name : {"config.json", "tokenizer.json", "tokenizer_config.json"})
        require_package_file(info, path, name, stage);
    // Only native auxiliary weights apply; another backend may share the directory.
    const auto files = info.find("files");
    if (files != info.end() && files->is_array()) {
        for (const auto& file : *files) {
            if (file.is_string() && std::filesystem::path(file.get<std::string>()).extension() == ".q4nx")
                require_package_file(info, path, file.get<std::string>(), stage);
        }
    }
}

}  // namespace flm::backend
