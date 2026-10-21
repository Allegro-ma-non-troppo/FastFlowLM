/// \file test_no_exit.cpp
/// \brief Load-path failures throw instead of terminating the process.
/// \note Before the change these conditions called exit(1) from inside the
///       constructor, so no caller could catch them. Every case here expects an
///       exception with a message and then goes on; reaching the end of main is
///       the assertion that the process survived. Nothing here needs an NPU: the
///       model is never loaded, only its catalog, config files and tokenizer are read.
#include "model_list.hpp"
#include "lm_config.hpp"
#include "../phi4_core/test_support.hpp"

#include <filesystem>
#include <utility>
#include <fstream>
#include <stdexcept>
#include <string>

#ifdef FLM_TEST_TOKENIZER
#include "tokenizer/tokenizer.hpp"
#endif

#ifdef _WIN32
#include <stdlib.h>
#else
#include <cstdlib>
#endif

namespace fs = std::filesystem;

namespace {

/// \brief a fresh scratch directory, empty
fs::path fresh_dir(const char* name) {
    const fs::path dir = fs::temp_directory_path() / "flm_no_exit_test" / name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir;
}

/// \brief run a call that must throw a std::exception with a non-empty message
template <typename Callable>
std::string expect_throw(Callable&& call) {
    const std::string message = RequireThrows<std::exception>(std::forward<Callable>(call));
    TEST_REQUIRE(!message.empty());
    return message;
}

void write_text(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out << text;
}

void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

/// \brief a catalog containing one entry, written to dir/model_list.json
fs::path write_catalog(const fs::path& dir, const nlohmann::json& entry) {
    const fs::path path = dir / "model_list.json";
    write_text(path, nlohmann::json{
        {"model_path", "models"},
        {"models", {{"demo", {{"1b", entry}}}}}}.dump(2));
    return path;
}

void test_missing_catalog_throws() {
    const fs::path dir = fresh_dir("missing_catalog");
    std::string list_path = (dir / "does_not_exist.json").string();
    std::string exe_dir = dir.string();
    const std::string message = expect_throw([&] {
        model_list models(list_path, exe_dir, "aie2p", {"flm"});
    });
    RequireContains(message, "Failed to open config file");
    RequireContains(message, "does_not_exist.json");
}

void test_unparsable_catalog_throws() {
    // Not a converted site (nlohmann already threw); kept as the load path's other failure.
    const fs::path dir = fresh_dir("unparsable_catalog");
    const fs::path path = dir / "model_list.json";
    write_text(path, "{ this is not json");
    std::string list_path = path.string();
    std::string exe_dir = dir.string();
    (void)expect_throw([&] {
        model_list models(list_path, exe_dir, "aie2p", {"flm"});
    });
}

void test_catalog_without_model_path_throws() {
    const fs::path dir = fresh_dir("no_model_path");
    const fs::path path = dir / "model_list.json";
    write_text(path, R"({"models":{}})");
    std::string list_path = path.string();
    std::string exe_dir = dir.string();
    (void)expect_throw([&] {
        model_list models(list_path, exe_dir, "aie2p", {"flm"});
    });
}

void test_supported_platforms_wrong_type_throws() {
    // A field of the wrong JSON type: a string where an array is required.
    const fs::path dir = fresh_dir("platforms_string");
    const fs::path path = write_catalog(dir, {{"name", "Demo"}, {"supported_platforms", "aie2p"}});
    std::string list_path = path.string();
    std::string exe_dir = dir.string();
    const std::string message = expect_throw([&] {
        model_list models(list_path, exe_dir, "aie2p", {"flm"});
    });
    RequireContains(message, "supported_platforms must be a non-empty array");
    RequireContains(message, "demo:1b");
}

void test_supported_platforms_empty_array_throws() {
    const fs::path dir = fresh_dir("platforms_empty");
    const fs::path path = write_catalog(dir, {{"name", "Demo"}, {"supported_platforms", nlohmann::json::array()}});
    std::string list_path = path.string();
    std::string exe_dir = dir.string();
    const std::string message = expect_throw([&] {
        model_list models(list_path, exe_dir, "aie2p", {"flm"});
    });
    RequireContains(message, "supported_platforms must be a non-empty array");
}

void test_supported_platforms_non_string_element_throws() {
    // A number inside the array. The first element is checked before any match,
    // so the error is reported even though "aie2p" appears later.
    const fs::path dir = fresh_dir("platforms_number");
    const fs::path path = write_catalog(dir, {{"name", "Demo"}, {"supported_platforms", nlohmann::json::array({2, "aie2p"})}});
    std::string list_path = path.string();
    std::string exe_dir = dir.string();
    const std::string message = expect_throw([&] {
        model_list models(list_path, exe_dir, "aie2p", {"flm"});
    });
    RequireContains(message, "supported_platforms must contain strings");
}

void test_empty_catalog_after_filtering_does_not_throw() {
    // The opposite case, which must stay a state and not an error: every entry is
    // for other silicon. The constructor reports it on stderr and returns.
    const fs::path dir = fresh_dir("filtered_empty");
    const fs::path path = write_catalog(dir, {{"name", "Demo"}, {"supported_platforms", nlohmann::json::array({"aie2p"})}});
    std::string list_path = path.string();
    std::string exe_dir = dir.string();
    model_list models(list_path, exe_dir, "dev04", {"flm"});
    TEST_REQUIRE(models.all_tags.empty());
}

/// \brief an LM_Config reads FLM_XCLBIN_PATH before it opens a model directory
void point_xclbin_at_fixture() {
    const fs::path root = fresh_dir("xclbin_root");
    fs::create_directories(root / "xclbins");
    set_env("FLM_XCLBIN_PATH", root.string());
}

void test_lm_config_missing_config_json_throws() {
    point_xclbin_at_fixture();
    const fs::path model_dir = fresh_dir("lm_config_empty_model");
    LM_Config config;
    const std::string message = expect_throw([&] {
        config.from_pretrained(model_dir.string());
    });
    RequireContains(message, "Failed to open file");
    RequireContains(message, "config.json");
}

void test_lm_config_missing_model_directory_throws() {
    point_xclbin_at_fixture();
    const fs::path model_dir = fs::temp_directory_path() / "flm_no_exit_test" / "no_such_model_dir";
    fs::remove_all(model_dir.parent_path() / "no_such_model_dir");
    LM_Config config;
    const std::string message = expect_throw([&] {
        config.from_pretrained(model_dir.string());
    });
    RequireContains(message, "Failed to open file");
}

void test_whisper_config_missing_config_json_throws() {
    point_xclbin_at_fixture();
    const fs::path model_dir = fresh_dir("whisper_config_empty_model");
    Whisper_Config config;
    const std::string message = expect_throw([&] {
        config.from_pretrained(model_dir.string());
    });
    RequireContains(message, "config.json");
}

void test_lm_config_malformed_config_json_throws() {
    // Not a converted site: nlohmann throws. Listed so the table of load failures is complete.
    point_xclbin_at_fixture();
    const fs::path model_dir = fresh_dir("lm_config_bad_json");
    write_text(model_dir / "config.json", "{ broken");
    LM_Config config;
    (void)expect_throw([&] {
        config.from_pretrained(model_dir.string());
    });
}

#ifdef FLM_TEST_TOKENIZER
void test_tokenizer_missing_json_throws() {
    // tokenizer.cpp: "Cannot open <dir>\tokenizer.json", previously exit(1).
    const fs::path model_dir = fresh_dir("tokenizer_empty_model");
    const std::string message = expect_throw([&] {
        Tokenizer tokenizer(model_dir.string());
    });
    RequireContains(message, "Cannot open");
    RequireContains(message, "tokenizer.json");
}
#endif

}  // namespace

int main() {
    RunTest(test_missing_catalog_throws, "model_list: missing config file throws");
    RunTest(test_unparsable_catalog_throws, "model_list: unparsable catalog throws");
    RunTest(test_catalog_without_model_path_throws, "model_list: catalog without model_path throws");
    RunTest(test_supported_platforms_wrong_type_throws,
            "model_list: supported_platforms of the wrong type throws");
    RunTest(test_supported_platforms_empty_array_throws,
            "model_list: empty supported_platforms throws");
    RunTest(test_supported_platforms_non_string_element_throws,
            "model_list: non-string supported_platforms element throws");
    RunTest(test_empty_catalog_after_filtering_does_not_throw,
            "model_list: catalog filtered to nothing does not throw");
    RunTest(test_lm_config_missing_config_json_throws,
            "LM_Config: model directory without config.json throws");
    RunTest(test_lm_config_missing_model_directory_throws,
            "LM_Config: missing model directory throws");
    RunTest(test_whisper_config_missing_config_json_throws,
            "Whisper_Config: model directory without config.json throws");
    RunTest(test_lm_config_malformed_config_json_throws,
            "LM_Config: malformed config.json throws");
#ifdef FLM_TEST_TOKENIZER
    RunTest(test_tokenizer_missing_json_throws, "Tokenizer: missing tokenizer.json throws");
#endif
    // Reaching this line is the point of the suite: the process is still alive
    // after every load failure above.
    if (FailureCount() == 0) {
        std::cout << "All no_exit tests passed; process survived every load-path failure\n";
        return 0;
    }
    return 1;  // RunTest's atexit hook forces the exit code and prints the count
}
