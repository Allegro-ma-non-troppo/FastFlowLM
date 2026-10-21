/// \file test_no_exit_automodel.cpp
/// \brief AutoModel load-path failures throw instead of calling exit(1).
/// \note Drives the real AutoModel::_shared_setup_tokenizer and
///       AutoModel::_shared_initialize_legacy_npu through a subclass that loads no
///       model. The tokenizer files are written into a temporary model directory. The
///       NPU case passes a null device, which is the condition that used to exit; no
///       device is opened in any case.
#include "AutoModel/automodel.hpp"
#include "../phi4_core/test_support.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace flm::backend {
// The real registry seeding lives in builtin_backends.cpp, which names every prebuilt
// engine library. This test links none of them, so the set is empty.
void register_builtin_backends(BackendRegistry&) {}
}  // namespace flm::backend

namespace {

/// \brief AutoModel with the two load steps exposed; nothing is loaded
class LoadProbe final : public AutoModel {
public:
    LoadProbe() : AutoModel(nullptr, "load-probe") {}

    // _shared_setup_tokenizer detects the chat format, which probes the tokenizer; a real
    // load builds the tokenizer first (_shared_initialize_model_state), so the probe does too
    // when the directory holds a tokenizer.json.
    nlohmann::json setup_tokenizer(const std::string& model_dir) {
        if (fs::exists(fs::path(model_dir) / "tokenizer.json")) {
            tokenizer = std::make_unique<Tokenizer>(model_dir);
        }
        return _shared_setup_tokenizer(model_dir);
    }
    void initialize_legacy_npu() { _shared_initialize_legacy_npu(false); }

    const std::vector<int>& eos_ids() const { return eos_token_ids; }
    int bos_id() const { return bos_token_id; }

    bool insert(chat_meta_info_t&, lm_uniform_input_t&, std::function<bool()>) override { return false; }
    std::string generate(chat_meta_info_t&, int, std::ostream&, std::function<bool()>) override { return {}; }
    std::string generate_with_prompt(chat_meta_info_t&, lm_uniform_input_t&, int, std::ostream&) override { return {}; }
    std::string apply_chat_template(nlohmann::ordered_json&, nlohmann::ordered_json) override { return {}; }
};

/// \brief a model directory holding the given tokenizer_config.json, or none when null
fs::path model_dir(const char* name, const char* tokenizer_config) {
    const fs::path dir = fs::temp_directory_path() / "flm_no_exit_automodel" / name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    if (tokenizer_config != nullptr) {
        std::ofstream(dir / "tokenizer_config.json", std::ios::binary) << tokenizer_config;
        // The smallest tokenizer.json the tokenizers library accepts.
        std::ofstream(dir / "tokenizer.json", std::ios::binary)
            << R"({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],)"
               R"("normalizer":null,"pre_tokenizer":null,"post_processor":null,)"
               R"("decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},)"
               R"("model":{"type":"BPE","vocab":{"a":0},"merges":[]}})";
    }
    return dir;
}

/// \brief the message of a std::exception that must be thrown, and must not be empty
template <typename Callable>
std::string expect_throw(Callable&& call) {
    const std::string message = RequireThrows<std::exception>(std::forward<Callable>(call));
    TEST_REQUIRE(!message.empty());
    return message;
}

void test_missing_tokenizer_config_throws() {
    const fs::path dir = model_dir("no_tokenizer_config", nullptr);
    LoadProbe probe;
    const std::string message = expect_throw([&] { probe.setup_tokenizer(dir.string()); });
    RequireContains(message, "Cannot open");
    RequireContains(message, "tokenizer_config.json");
}

void test_no_template_file_and_no_chat_template_throws() {
    const fs::path dir = model_dir("no_template",
        R"({"bos_token": null, "eos_token": "</s>", "eos_token_id": [2]})");
    LoadProbe probe;
    const std::string message = expect_throw([&] { probe.setup_tokenizer(dir.string()); });
    RequireContains(message, "No template file found");
}

void test_null_eos_token_throws() {
    const fs::path dir = model_dir("eos_null",
        R"({"bos_token": null, "eos_token": null, "chat_template": "x", "eos_token_id": [2]})");
    LoadProbe probe;
    const std::string message = expect_throw([&] { probe.setup_tokenizer(dir.string()); });
    RequireContains(message, "eos_token is null");
}

void test_missing_bos_token_id_throws() {
    // bos_token is set, so bos_token_id must be an integer. No backend forces it here.
    const fs::path dir = model_dir("bos_without_id",
        R"({"bos_token": "<s>", "eos_token": "</s>", "chat_template": "x", "eos_token_id": [2]})");
    LoadProbe probe;
    const std::string message = expect_throw([&] { probe.setup_tokenizer(dir.string()); });
    RequireContains(message, "bos_token_id is missing or not an integer");
}

void test_eos_token_id_not_an_array_throws() {
    const fs::path dir = model_dir("eos_id_not_array",
        R"({"bos_token": null, "eos_token": "</s>", "chat_template": "x", "eos_token_id": 2})");
    LoadProbe probe;
    const std::string message = expect_throw([&] { probe.setup_tokenizer(dir.string()); });
    RequireContains(message, "eos_token_id is missing or not an array");
}

void test_eos_token_id_with_a_non_integer_throws() {
    const fs::path dir = model_dir("eos_id_string",
        R"({"bos_token": null, "eos_token": "</s>", "chat_template": "x", "eos_token_id": ["two"]})");
    LoadProbe probe;
    const std::string message = expect_throw([&] { probe.setup_tokenizer(dir.string()); });
    RequireContains(message, "must be an array of integers");
}

void test_unreadable_chat_template_file_throws() {
    // chat_template.jinja exists but cannot be opened for reading: it is a directory here.
    const fs::path dir = model_dir("template_is_directory",
        R"({"bos_token": null, "eos_token": "</s>", "chat_template": "x", "eos_token_id": [2]})");
    fs::create_directories(dir / "chat_template.jinja");
    LoadProbe probe;
    const std::string message = expect_throw([&] { probe.setup_tokenizer(dir.string()); });
    RequireContains(message, "Cannot open");
    RequireContains(message, "chat_template.jinja");
}

void test_null_npu_device_throws() {
    // The device is the one thing an NPU load needs; a null device used to exit(1).
    LoadProbe probe;
    const std::string message = expect_throw([&] { probe.initialize_legacy_npu(); });
    RequireContains(message, "NPU device instance is nullptr");
}

void test_valid_tokenizer_config_loads_without_throwing() {
    // The happy path, so the cases above are not passing because every load fails.
    const fs::path dir = model_dir("valid",
        R"({"bos_token": "<s>", "bos_token_id": 1, "eos_token": "</s>", "eos_token_id": [2], "chat_template": "{{ messages[0].content }}"})");
    LoadProbe probe;
    probe.setup_tokenizer(dir.string());
    TEST_REQUIRE(probe.bos_id() == 1);
    // The configured stop id comes first; chat-format detection may append tool-call markers.
    TEST_REQUIRE(!probe.eos_ids().empty() && probe.eos_ids().front() == 2);
}

}  // namespace

int main() {
    RunTest(test_missing_tokenizer_config_throws, "AutoModel: missing tokenizer_config.json throws");
    RunTest(test_no_template_file_and_no_chat_template_throws,
            "AutoModel: no chat template file and no chat_template throws");
    RunTest(test_null_eos_token_throws, "AutoModel: null eos_token throws");
    RunTest(test_missing_bos_token_id_throws, "AutoModel: bos_token without bos_token_id throws");
    RunTest(test_eos_token_id_not_an_array_throws, "AutoModel: eos_token_id that is not an array throws");
    RunTest(test_eos_token_id_with_a_non_integer_throws, "AutoModel: non-integer eos_token_id throws");
    RunTest(test_unreadable_chat_template_file_throws,
            "AutoModel: unreadable chat_template.jinja throws");
    RunTest(test_null_npu_device_throws, "AutoModel: null NPU device throws (no device opened)");
    RunTest(test_valid_tokenizer_config_loads_without_throwing,
            "AutoModel: valid tokenizer_config.json loads without throwing");
    if (FailureCount() == 0) {
        std::cout << "All automodel no_exit tests passed; process survived every load-path failure\n";
        return 0;
    }
    return 1;  // RunTest's atexit hook forces the exit code and prints the count
}
