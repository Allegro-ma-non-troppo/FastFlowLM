/// \file backend_package.hpp
/// \brief Lightweight package file checks shared by backend preflight validators
#pragma once

#include "AutoModel/model_backend.hpp"

#include <filesystem>
#include <stdexcept>

namespace flm::backend {

/// \brief require a loader entry point, without rejecting an incomplete legacy manifest
/// \note Local files take precedence. Before download, only a manifest naming a
///       recognized weight format can establish that a different entry is missing.
inline void require_package_file(const nlohmann::json& info,
                                 const std::filesystem::path& root,
                                 const std::string& filename,
                                 PackageValidationStage stage) {
    if (std::filesystem::is_regular_file(root / filename)) return;
    if (stage == PackageValidationStage::Manifest) {
        const auto files = info.find("files");
        if (files == info.end() || !files->is_array()) return;
        bool names_weights = false;
        for (const auto& file : *files) {
            if (!file.is_string()) continue;
            const auto name = file.get<std::string>();
            if (name == filename) return;
            const auto extension = std::filesystem::path(name).extension().string();
            names_weights |= extension == ".q4nx" || extension == ".gguf" ||
                             name == "genai_config.json";
        }
        if (!names_weights) return;
        throw std::runtime_error("The package manifest does not provide " + filename);
    }
    throw std::runtime_error("Missing required file " + (root / filename).string());
}

}  // namespace flm::backend
