/// \file test_resource_root.cpp
/// \brief Resource root and models directory overrides in utils.
/// \note Exercises utils::set_resource_root, utils::set_models_directory and the
///       lookups that consult them (find_model_list, find_model_info,
///       find_xclbin_path, get_models_directory). Only utils.cpp is linked: no NPU
///       runtime, no XRT, no engine libraries, so it runs on any Windows host.
///       Every test resets the process-wide state it touches, so the cases can run
///       in any order.
#include "utils/utils.hpp"
#include "../phi4_core/test_support.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <process.h>
#endif

namespace fs = std::filesystem;

namespace {

const char* const kConfigEnv = "FLM_CONFIG_PATH";
const char* const kModelInfoEnv = "FLM_MODELINFO_PATH";
const char* const kXclbinEnv = "FLM_XCLBIN_PATH";
const char* const kModelPathEnv = "FLM_MODEL_PATH";

/// \brief remove an environment variable for this process (the CRT copy that std::getenv reads)
void unset_env(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

/// \brief the state every test starts from: no root, no models directory, no FLM_* overrides
void reset_state() {
    utils::set_resource_root("");
    utils::set_models_directory("");
    unset_env(kConfigEnv);
    unset_env(kModelInfoEnv);
    unset_env(kXclbinEnv);
    unset_env(kModelPathEnv);
}

/// \brief a per-run scratch area, recreated empty
fs::path scratch_area() {
    const fs::path base = fs::temp_directory_path() / "flm_resource_root_test";
    std::error_code ec;
    fs::remove_all(base, ec);
    fs::create_directories(base);
    return base;
}

/// \brief put a fake model_list.json, model_info.json and xclbins/ into a directory
void populate(const fs::path& dir) {
    fs::create_directories(dir / "xclbins");
    std::ofstream(dir / "model_list.json") << R"({"model_path":"models","models":{}})";
    std::ofstream(dir / "model_info.json") << "{}";
}

/// \brief true when two paths name the same location, ignoring separator style and "./"
bool same_path(const std::string& a, const fs::path& b) {
    return fs::path(a).lexically_normal() == b.lexically_normal();
}

/// \brief true when the path names an entry directly inside dir
bool directly_inside(const std::string& path, const fs::path& dir) {
    return fs::path(path).lexically_normal().parent_path() == dir.lexically_normal();
}

/// \brief the working directory for the duration of a test
class ScopedCwd {
public:
    explicit ScopedCwd(const fs::path& dir) : saved_(fs::current_path()) { fs::current_path(dir); }
    ~ScopedCwd() { fs::current_path(saved_); }
private:
    fs::path saved_;
};

/// \brief a root with files, and a decoy elsewhere with the same names
struct Layout {
    fs::path root;
    fs::path decoy;
    fs::path empty_root;
};

Layout make_layout() {
    const fs::path base = scratch_area();
    Layout layout{base / "root", base / "decoy", base / "empty_root"};
    populate(layout.root);
    populate(layout.decoy);
    fs::create_directories(layout.empty_root);
    return layout;
}

void test_env_paths_still_win_under_a_root() {
    // The documented precedence (src/include/utils/utils.hpp, set_resource_root):
    // FLM_CONFIG_PATH, FLM_MODELINFO_PATH and FLM_XCLBIN_PATH are consulted first.
    reset_state();
    const Layout layout = make_layout();
    utils::set_resource_root(layout.root.string());
    set_env(kConfigEnv, (layout.decoy / "model_list.json").string());
    set_env(kModelInfoEnv, (layout.decoy / "model_info.json").string());
    set_env(kXclbinEnv, layout.decoy.string());

    TEST_REQUIRE(same_path(utils::find_model_list(), layout.decoy / "model_list.json"));
    TEST_REQUIRE(same_path(utils::find_model_info(), layout.decoy / "model_info.json"));
    TEST_REQUIRE(same_path(utils::find_xclbin_path(), layout.decoy));
    reset_state();
}

void test_rooted_lookups_resolve_inside_root_without_env() {
    // With no FLM_* variables, a rooted lookup finds the files in the root even
    // when the working directory holds other copies of them.
    reset_state();
    const Layout layout = make_layout();
    utils::set_resource_root(layout.root.string());
    ScopedCwd cwd(layout.decoy);

    TEST_REQUIRE(same_path(utils::get_resource_directory(), layout.root));
    const std::string list_path = utils::find_model_list();
    TEST_REQUIRE(same_path(fs::path(list_path).parent_path().string(), layout.root));
    TEST_REQUIRE(fs::equivalent(list_path, layout.root / "model_list.json"));
    const std::string info_path = utils::find_model_info();
    TEST_REQUIRE(fs::equivalent(info_path, layout.root / "model_info.json"));
    TEST_REQUIRE(same_path(utils::find_xclbin_path(), layout.root));
    reset_state();
}

void test_rooted_lookup_of_missing_file_throws() {
    // The root is empty, the working directory holds a complete copy. A rooted
    // lookup must not fall back to the working directory or the executable directory.
    reset_state();
    const Layout layout = make_layout();
    utils::set_resource_root(layout.empty_root.string());
    ScopedCwd cwd(layout.decoy);

    const std::string list_message = RequireThrows<std::runtime_error>([] { utils::find_model_list(); });
    TEST_REQUIRE(!list_message.empty());
    const std::string info_message = RequireThrows<std::runtime_error>([] { utils::find_model_info(); });
    TEST_REQUIRE(!info_message.empty());
    const std::string xclbin_message = RequireThrows<std::runtime_error>([] { utils::find_xclbin_path(); });
    TEST_REQUIRE(!xclbin_message.empty());
    reset_state();
}

void test_rooted_lookup_reads_the_bundle_beside_the_root() {
    // Like the executable directory, a root also has the relocatable bundle layout
    // <root>/../share/flm (src/include/utils/utils.hpp, set_resource_root).
    reset_state();
    const Layout layout = make_layout();
    const fs::path bundle = layout.empty_root.parent_path() / "share" / "flm";
    populate(bundle);
    utils::set_resource_root(layout.empty_root.string());

    TEST_REQUIRE(fs::equivalent(utils::find_model_list(), bundle / "model_list.json"));
#ifndef _WIN32
    // On Windows find_model_info only looks next to the executable (or the root).
    TEST_REQUIRE(fs::equivalent(utils::find_model_info(), bundle / "model_info.json"));
#endif
    TEST_REQUIRE(same_path(utils::find_xclbin_path(), bundle));
    fs::remove_all(bundle);
    reset_state();
}
void test_unset_root_keeps_env_precedence() {
    // Without a root nothing changes: the variable wins, and the resource
    // directory is the executable directory.
    reset_state();
    const Layout layout = make_layout();
    set_env(kConfigEnv, (layout.decoy / "model_list.json").string());
    set_env(kModelInfoEnv, (layout.decoy / "model_info.json").string());
    set_env(kXclbinEnv, layout.decoy.string());

    TEST_REQUIRE(same_path(utils::find_model_list(), layout.decoy / "model_list.json"));
    TEST_REQUIRE(same_path(utils::find_model_info(), layout.decoy / "model_info.json"));
    TEST_REQUIRE(same_path(utils::find_xclbin_path(), layout.decoy));
    TEST_REQUIRE(utils::get_resource_directory() == utils::get_executable_directory());
    reset_state();
}

void test_set_and_clear_resource_root() {
    reset_state();
    const Layout layout = make_layout();
    utils::set_resource_root(layout.root.string());
    TEST_REQUIRE(same_path(utils::get_resource_directory(), layout.root));
    utils::set_resource_root("");
    TEST_REQUIRE(utils::get_resource_directory() == utils::get_executable_directory());
    reset_state();
}

void test_models_directory_override() {
    reset_state();
    const Layout layout = make_layout();
    const std::string custom = (layout.root / "models").string();

    utils::set_models_directory(custom);
    TEST_REQUIRE(utils::get_models_directory() == custom);

    // An empty value restores the default, which is not the custom directory.
    utils::set_models_directory("");
    const std::string restored = utils::get_models_directory();
    TEST_REQUIRE(restored != custom);
    TEST_REQUIRE(!restored.empty());
    reset_state();
}

void test_models_directory_env_wins_over_setter() {
    // get_models_directory still prefers FLM_MODEL_PATH over the value set by the host.
    reset_state();
    const Layout layout = make_layout();
    const std::string custom = (layout.root / "models").string();
    const std::string from_env = (layout.decoy / "env_models").string();

    utils::set_models_directory(custom);
    set_env(kModelPathEnv, from_env);
    TEST_REQUIRE(utils::get_models_directory() == from_env);
    reset_state();
}

}  // namespace

int main() {
    RunTest(test_env_paths_still_win_under_a_root,
            "FLM_* env paths still win under a root (documented)");
    RunTest(test_rooted_lookups_resolve_inside_root_without_env,
            "rooted lookups resolve inside the root without env");
    RunTest(test_rooted_lookup_of_missing_file_throws,
            "rooted lookup of a missing file throws");
    RunTest(test_rooted_lookup_reads_the_bundle_beside_the_root,
            "rooted lookup reads the share/flm bundle beside the root");
    RunTest(test_unset_root_keeps_env_precedence,
            "unset root keeps env precedence and exe directory");
    RunTest(test_set_and_clear_resource_root, "set and clear resource root");
    RunTest(test_models_directory_override, "models directory override");
    RunTest(test_models_directory_env_wins_over_setter,
            "FLM_MODEL_PATH wins over set_models_directory");
    if (FailureCount() == 0) {
        std::cout << "All resource_root tests passed\n";
        return 0;
    }
    return 1;  // RunTest's atexit hook forces the exit code and prints the count
}
