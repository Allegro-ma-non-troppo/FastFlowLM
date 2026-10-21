/// \file test_chat_session.cpp
/// \brief flm::request::ChatSession driven through a fake AutoModel.
/// \note The engine is a subclass of AutoModel that overrides the pure virtuals
///       (insert, generate, generate_with_prompt, apply_chat_template) and the
///       virtual hooks the session uses (clear_context, get_current_context_length,
///       configure_parameter). AutoModel's own code is the real one: the decoding
///       constraint, the sampler setters and the chat format are not faked.
///       No model is loaded and no NPU is touched. Two things are synthetic and are
///       said so where they are used: the vocabulary (a byte-level tokenizer.json,
///       written by this test) and the tokens the engine emits (scripted text).
///       Non-ASCII text is spelled with \x escapes so the source stays ASCII.
#include "request/chat_request.hpp"
#include "AutoModel/automodel.hpp"
#include "chat/chat_format.hpp"
#include "chat/chat_msg.hpp"
#include "lm_config.hpp"
#include "modules/sampler.hpp"
#include "../phi4_core/test_support.hpp"

#include <any>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

namespace flm::backend {
// The real registry seeding lives in builtin_backends.cpp and names every prebuilt
// engine library, which this test deliberately does not link. Nothing here registers
// a backend, so the empty function is enough for model_backend.cpp.
void register_builtin_backends(BackendRegistry&) {}
}  // namespace flm::backend

namespace {

/// \brief vocabulary size of the fake model; the synthetic tokenizer has exactly this many tokens
constexpr int kVocab = 256;

const std::function<bool()> kNeverCancelled = [] { return false; };

/// \brief the tokenizer.json written for the grammar engine
/// \note A byte-level BPE vocabulary with one token per byte, in GPT-2's byte-to-character
///       mapping, and no merges. It is the smallest file xgrammar accepts; it is not
///       a model tokenizer, and the test never decodes or samples with it.
std::string synthetic_tokenizer_json() {
    // GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to U+0100 and up.
    std::vector<int> bytes;
    for (int b = 0x21; b <= 0x7E; ++b) bytes.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bytes.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bytes.push_back(b);
    std::map<int, int> codepoint;
    for (int b : bytes) codepoint[b] = b;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        if (codepoint.count(b) == 0) codepoint[b] = 256 + extra++;
    }
    auto utf8 = [](int cp) {
        std::string out;
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return out;
    };
    json vocab = json::object();
    for (int b = 0; b < 256; ++b) {
        vocab[utf8(codepoint[b])] = b;
    }
    json tokenizer;
    tokenizer["version"] = "1.0";
    tokenizer["added_tokens"] = json::array();
    tokenizer["pre_tokenizer"] = {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", true}};
    tokenizer["decoder"] = {{"type", "ByteLevel"}, {"add_prefix_space", true}, {"trim_offsets", true}, {"use_regex", true}};
    tokenizer["model"] = {{"type", "BPE"}, {"vocab", vocab}, {"merges", json::array()}};
    return tokenizer.dump();
}

/// \brief a directory holding the synthetic tokenizer.json, recreated each run
fs::path model_directory() {
    const fs::path dir = fs::temp_directory_path() / "flm_chat_session_test";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    std::ofstream(dir / "tokenizer.json", std::ios::binary) << synthetic_tokenizer_json();
    return dir;
}

/// \brief AutoModel with the engine behaviour scripted by the test
class FakeEngine final : public AutoModel {
public:
    explicit FakeEngine(const fs::path& dir) : AutoModel(nullptr, "fake-model") {
        model_path = dir.string();
        MAX_L = 4096;
        chat_format = chat_format_make(chat_format_kind::CONTENT_ONLY, false);
        lm_config = std::make_unique<LM_Config>();
        lm_config->_json_config["vocab_size"] = kVocab;
        sampler_config defaults;
        set_sampler(defaults);
    }

    // ---- scripted behaviour -------------------------------------------------------
    std::string reply;                                   // text generate() writes and returns
    stop_reason_t reply_stop = EOT_DETECTED;             // stop reason generate() reports
    std::function<void(std::ostream&)> stream_script;    // replaces the reply when set
    std::function<void()> generate_fault;                // runs first in generate(); may throw
    bool insert_result = true;                           // what insert() returns
    bool insert_throws = false;                          // insert() throws instead
    int prompt_tokens = 12;

    // ---- observations ---------------------------------------------------------------
    int insert_calls = 0;
    int generate_calls = 0;
    int clear_calls = 0;
    bool last_restore_allowed = false;
    std::vector<std::pair<std::string, std::any>> configured;

    const Sampler& sampler_state() const { return *sampler; }
    void use_format(chat_format_kind kind, bool with_reasoning) {
        chat_format = chat_format_make(kind, with_reasoning);
    }

    // ---- pure virtuals ------------------------------------------------------------------
    bool insert(chat_meta_info_t& meta, lm_uniform_input_t& input, std::function<bool()> is_cancelled) override {
        (void)input;
        (void)is_cancelled;
        ++insert_calls;
        last_restore_allowed = meta.restore_allowed;
        if (insert_throws) throw std::runtime_error("insert failed");
        meta.prompt_tokens = prompt_tokens;
        meta.prompt_tail.clear();
        context_tokens_ = prompt_tokens;
        return insert_result;
    }

    std::string generate(chat_meta_info_t& meta, int length_limit, std::ostream& os,
                         std::function<bool()> is_cancelled) override {
        (void)length_limit;
        (void)is_cancelled;
        ++generate_calls;
        if (generate_fault) generate_fault();
        meta.stop_reason = reply_stop;
        meta.generated_tokens = 7;
        if (stream_script) {
            stream_script(os);
            return reply;
        }
        os << reply;
        return reply;
    }

    std::string generate_with_prompt(chat_meta_info_t& meta, lm_uniform_input_t& input, int length_limit,
                                     std::ostream& os) override {
        (void)meta;
        (void)input;
        (void)length_limit;
        os << reply;
        return reply;
    }

    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) override {
        (void)messages;
        (void)tools;
        return {};
    }

    // ---- virtual hooks the session uses ----------------------------------------------
    // The base clear_context() reaches into lm_engine, which this fake does not have.
    void clear_context() override {
        ++clear_calls;
        context_tokens_ = 0;
    }

    int get_current_context_length() override { return context_tokens_; }

    bool configure_parameter(std::string parameter_name, const std::any& value) override {
        configured.emplace_back(std::move(parameter_name), value);
        return true;
    }

private:
    int context_tokens_ = 0;
};

/// \brief collects the deltas a streamed request delivers
class RecordingSink final : public flm::request::ChatSink {
public:
    void on_delta(const chat_msg_delta& delta) override { deltas.push_back(delta); }

    std::string content() const {
        std::string text;
        for (const auto& d : deltas) text += d.content;
        return text;
    }

    std::vector<chat_msg_delta> deltas;
};

json message(const char* role, const std::string& content) {
    return json{{"role", role}, {"content", content}};
}

flm::request::ChatRequest make_request(const json& messages, bool stream = false,
                                       const std::string& model = "test-model") {
    flm::request::ChatRequest request;
    request.model = model;
    request.messages = messages;
    request.stream = stream;
    return request;
}

flm::request::ChatOutcome run(flm::request::ChatSession& session, FakeEngine& engine,
                              const flm::request::ChatRequest& request, RecordingSink* sink = nullptr,
                              const std::function<bool()>& is_cancelled = kNeverCancelled) {
    return session.run_chat(engine, request, sink, is_cancelled);
}

/// \brief true when every byte sequence in the string is complete UTF-8
bool valid_utf8(const std::string& text) {
    size_t i = 0;
    while (i < text.size()) {
        const auto lead = static_cast<unsigned char>(text[i]);
        size_t length = 0;
        if (lead < 0x80) length = 1;
        else if ((lead & 0xE0) == 0xC0) length = 2;
        else if ((lead & 0xF0) == 0xE0) length = 3;
        else if ((lead & 0xF8) == 0xF0) length = 4;
        if (length == 0 || i + length > text.size()) return false;
        for (size_t k = 1; k < length; ++k) {
            if ((static_cast<unsigned char>(text[i + k]) & 0xC0) != 0x80) return false;
        }
        i += length;
    }
    return true;
}

/// \brief the messages of the multi-turn cache cases
json three_messages() {
    return json::array({message("user", "hi"), message("assistant", "hello"), message("user", "how are you")});
}

json five_messages() {
    json messages = three_messages();
    messages.push_back(message("assistant", "fine"));
    messages.push_back(message("user", "bye"));
    return messages;
}

// ---------------------------------------------------------------------------------------

void test_non_stream_completion_returns_parsed_message_and_meta() {
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.reply = "Hello there.";

    auto outcome = run(session, engine, make_request(json::array({message("user", "hi")})));
    TEST_REQUIRE(outcome.end == flm::request::ChatEnd::completed);
    TEST_REQUIRE(outcome.message.content == "Hello there.");
    TEST_REQUIRE(outcome.message.reasoning.empty());
    TEST_REQUIRE(outcome.message.tool_calls.empty());
    TEST_REQUIRE(outcome.meta.stop_reason == EOT_DETECTED);
    TEST_REQUIRE(outcome.meta.prompt_tokens == engine.prompt_tokens);
    TEST_REQUIRE(outcome.meta.generated_tokens == 7);
    TEST_REQUIRE(outcome.active_kv_tokens == engine.prompt_tokens);
    TEST_REQUIRE(outcome.max_kv_tokens == 4096);
    TEST_REQUIRE(engine.generate_calls == 1);
    // The first request on a fresh session misses the prompt cache, which clears the
    // (empty) engine context once.
    TEST_REQUIRE(engine.clear_calls == 1);
}

void test_stream_delivers_whole_utf8_characters() {
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    RecordingSink sink;
    // "h", U+00E9, "llo ", U+4E2D split across two writes with a flush in between,
    // then " done.". The partial lead bytes must not reach the sink.
    const std::string full = "h\xC3\xA9llo \xE4\xB8\xAD done.";
    std::string mid_text;
    bool checked_mid = false;
    engine.stream_script = [&](std::ostream& os) {
        os << "h\xC3\xA9llo ";
        os.flush();
        os.write("\xE4\xB8", 2);
        os.flush();
        mid_text = sink.content();
        checked_mid = true;
        os.write("\xAD", 1);
        os.flush();
        os << " done.";
    };

    auto outcome = run(session, engine, make_request(json::array({message("user", "hi")}), true), &sink);
    TEST_REQUIRE(checked_mid);
    TEST_REQUIRE(outcome.end == flm::request::ChatEnd::completed);
    TEST_REQUIRE(outcome.message.content == full);
    TEST_REQUIRE(sink.content() == full);
    TEST_REQUIRE(!sink.deltas.empty());
    for (const auto& delta : sink.deltas) {
        TEST_REQUIRE(valid_utf8(delta.content));
        TEST_REQUIRE(delta.reasoning.empty());
        TEST_REQUIRE(!delta.is_tool_call());
    }
    // Before the last byte arrived, what the sink had is a whole-character prefix of the answer.
    TEST_REQUIRE(valid_utf8(mid_text));
    TEST_REQUIRE(full.compare(0, mid_text.size(), mid_text) == 0);
    TEST_REQUIRE(mid_text.find("\xE4") == std::string::npos);
}

void test_stream_without_sink_is_rejected() {
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.reply = "unused";
    const std::string message_text = RequireThrows<std::invalid_argument>([&] {
        run(session, engine, make_request(json::array({message("user", "hi")}), true), nullptr);
    });
    RequireContains(message_text, "sink");
    TEST_REQUIRE(engine.insert_calls == 0);
}

void test_prompt_cache_reuses_an_extended_prompt() {
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.reply = "ok";

    auto first = run(session, engine, make_request(three_messages()));
    TEST_REQUIRE(first.meta.restore_allowed == false);
    TEST_REQUIRE(engine.last_restore_allowed == false);
    const int clears_after_first = engine.clear_calls;

    auto second = run(session, engine, make_request(five_messages()));
    TEST_REQUIRE(engine.last_restore_allowed == true);
    TEST_REQUIRE(second.meta.restore_allowed == true);
    TEST_REQUIRE(engine.clear_calls == clears_after_first);  // no clear on a hit
}

void test_prompt_cache_misses_when_the_first_message_changed() {
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.reply = "ok";
    run(session, engine, make_request(three_messages()));
    const int clears = engine.clear_calls;

    json changed = five_messages();
    changed[0]["content"] = "CHANGED";
    auto second = run(session, engine, make_request(changed));
    TEST_REQUIRE(engine.last_restore_allowed == false);
    TEST_REQUIRE(second.meta.restore_allowed == false);
    TEST_REQUIRE(engine.clear_calls == clears + 1);
}

void test_prompt_cache_is_not_used_when_the_request_opts_out() {
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.reply = "ok";
    run(session, engine, make_request(three_messages()));
    const int clears = engine.clear_calls;

    auto request = make_request(five_messages());
    request.allow_prompt_cache = false;
    auto second = run(session, engine, request);
    TEST_REQUIRE(engine.last_restore_allowed == false);
    TEST_REQUIRE(second.meta.restore_allowed == false);
    TEST_REQUIRE(engine.clear_calls == clears + 1);
}

void test_prompt_cache_reset_forces_a_miss() {
    // reset() clears the stored checksums and un-primes the cache, so the next request
    // misses even when it extends the conversation seen before the reset. Without the
    // reset the same request is a hit (see test_prompt_cache_reuses_an_extended_prompt).
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.reply = "ok";
    run(session, engine, make_request(three_messages()));
    session.reset_prompt_cache();

    auto after_reset = run(session, engine, make_request(five_messages()));
    TEST_REQUIRE(after_reset.meta.restore_allowed == false);
    TEST_REQUIRE(engine.last_restore_allowed == false);
}
void test_insert_refused_by_cancellation_is_prefill_cancelled() {
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.insert_result = false;
    const int clears = engine.clear_calls;

    auto outcome = run(session, engine, make_request(json::array({message("user", "hi")})), nullptr,
                       [] { return true; });
    TEST_REQUIRE(outcome.end == flm::request::ChatEnd::prefill_cancelled);
    TEST_REQUIRE(outcome.meta.stop_reason == CANCEL_DETECTED);
        // The first request on a fresh session misses the prompt cache and clears the (empty)
    // context; the refusal or failure then clears it once more.
    TEST_REQUIRE(engine.clear_calls == clears + 2);
    TEST_REQUIRE(engine.generate_calls == 0);
}

void test_insert_refused_without_cancellation_is_context_exceeded() {
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.insert_result = false;
    const int clears = engine.clear_calls;

    auto outcome = run(session, engine, make_request(json::array({message("user", "hi")})));
    TEST_REQUIRE(outcome.end == flm::request::ChatEnd::context_exceeded);
        // The first request on a fresh session misses the prompt cache and clears the (empty)
    // context; the refusal or failure then clears it once more.
    TEST_REQUIRE(engine.clear_calls == clears + 2);
    TEST_REQUIRE(engine.generate_calls == 0);
    TEST_REQUIRE(outcome.active_kv_tokens == engine.prompt_tokens);
}

void test_runtime_error_from_generate_clears_and_rethrows() {
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.generate_fault = [] { throw std::runtime_error("engine broke"); };
    const int clears = engine.clear_calls;

    const std::string text = RequireThrows<std::runtime_error>([&] {
        run(session, engine, make_request(json::array({message("user", "hi")})));
    });
    RequireContains(text, "engine broke");
        // The first request on a fresh session misses the prompt cache and clears the (empty)
    // context; the refusal or failure then clears it once more.
    TEST_REQUIRE(engine.clear_calls == clears + 2);
}

void test_runtime_error_from_streamed_generate_clears_and_rethrows() {
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    RecordingSink sink;
    engine.generate_fault = [] { throw std::runtime_error("stream broke"); };
    const int clears = engine.clear_calls;

    const std::string text = RequireThrows<std::runtime_error>([&] {
        run(session, engine, make_request(json::array({message("user", "hi")}), true), &sink);
    });
    RequireContains(text, "stream broke");
        // The first request on a fresh session misses the prompt cache and clears the (empty)
    // context; the refusal or failure then clears it once more.
    TEST_REQUIRE(engine.clear_calls == clears + 2);
}

void test_streamed_model_error_without_session_clear_leaves_context_to_engine() {
    // In streaming mode the engine owns the cleanup for this error (session_cleared == false),
    // so the session must not clear the context and must rethrow the same exception.
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    RecordingSink sink;
    engine.generate_fault = [] { throw ModelRequestError(500, false, "engine refused"); };
    const int clears = engine.clear_calls;

    bool rethrown = false;
    try {
        run(session, engine, make_request(json::array({message("user", "hi")}), true), &sink);
    } catch (const ModelRequestError& error) {
        rethrown = true;
        TEST_REQUIRE(error.http_code() == 500);
        TEST_REQUIRE(!error.session_cleared());
        TEST_REQUIRE(std::string(error.what()) == "engine refused");
    }
    TEST_REQUIRE(rethrown);
    // Only the cache-miss clear of the first request on a fresh session; the failure adds none.
    TEST_REQUIRE(engine.clear_calls == clears + 1);
}

void test_non_stream_model_error_clears_the_context_as_the_handler_did() {
    // Without a stream, the session cleans up after any exception from generate(),
    // ModelRequestError included. The pre-refactor handler did the same in its generic
    // std::exception branch, so this pins the behaviour rather than a new rule.
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.generate_fault = [] { throw ModelRequestError(400, false, "bad media"); };
    const int clears = engine.clear_calls;

    bool rethrown = false;
    try {
        run(session, engine, make_request(json::array({message("user", "hi")})));
    } catch (const ModelRequestError& error) {
        rethrown = true;
        TEST_REQUIRE(error.http_code() == 400);
        TEST_REQUIRE(std::string(error.what()) == "bad media");
    }
    TEST_REQUIRE(rethrown);
        // The first request on a fresh session misses the prompt cache and clears the (empty)
    // context; the refusal or failure then clears it once more.
    TEST_REQUIRE(engine.clear_calls == clears + 2);
}

void test_model_error_that_cleared_the_session_resets_the_cache() {
    // session_cleared == true: the session resets its prompt cache, so a request that
    // extends the failed one misses; with the cache kept it would be a hit (see the
    // counterpart below).
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    RecordingSink sink;
    engine.reply = "ok";
    run(session, engine, make_request(three_messages()));
    engine.generate_fault = [] { throw ModelRequestError(500, true, "cleared by engine"); };
    bool rethrown = false;
    try {
        run(session, engine, make_request(five_messages(), true), &sink);
    } catch (const ModelRequestError&) {
        rethrown = true;
    }
    TEST_REQUIRE(rethrown);
    engine.generate_fault = nullptr;
    json extended = five_messages();
    extended.push_back(message("assistant", "ok"));
    extended.push_back(message("user", "again"));
    auto after = run(session, engine, make_request(extended));
    TEST_REQUIRE(after.meta.restore_allowed == false);
}

void test_model_error_without_session_clear_keeps_the_cache() {
    // Counterpart: session_cleared == false leaves the cache alone, so a request that
    // extends the failed one is still a hit.
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    RecordingSink sink;
    engine.reply = "ok";
    run(session, engine, make_request(three_messages()));
    engine.generate_fault = [] { throw ModelRequestError(400, false, "rejected"); };
    bool rethrown = false;
    try {
        run(session, engine, make_request(five_messages(), true), &sink);
    } catch (const ModelRequestError&) {
        rethrown = true;
    }
    TEST_REQUIRE(rethrown);
    engine.generate_fault = nullptr;
    json extended = five_messages();
    extended.push_back(message("assistant", "ok"));
    extended.push_back(message("user", "again"));
    auto after = run(session, engine, make_request(extended));
    TEST_REQUIRE(after.meta.restore_allowed == true);
}
void test_single_turn_engine_rejects_history() {
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.single_turn = true;

    auto outcome = run(session, engine, make_request(three_messages()));
    TEST_REQUIRE(outcome.end == flm::request::ChatEnd::single_turn_rejected);
    TEST_REQUIRE(engine.insert_calls == 0);
    TEST_REQUIRE(engine.generate_calls == 0);
}

void test_single_turn_engine_accepts_one_user_message() {
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.single_turn = true;
    engine.reply = "one shot";

    auto outcome = run(session, engine, make_request(json::array({message("user", "hi")})));
    TEST_REQUIRE(outcome.end == flm::request::ChatEnd::completed);
    TEST_REQUIRE(outcome.message.content == "one shot");
    TEST_REQUIRE(engine.insert_calls == 1);
}

void test_tool_call_in_output_promotes_the_stop_reason() {
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.use_format(chat_format_kind::HERMES_JSON, false);
    engine.reply =
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    engine.reply_stop = EOT_DETECTED;

    auto request = make_request(json::array({message("user", "weather in Paris?")}));
    request.tools = json::array({json{
        {"type", "function"},
        {"function", {{"name", "get_weather"},
                      {"description", "Current weather for a city"},
                      {"parameters", {{"type", "object"},
                                      {"properties", {{"city", {{"type", "string"}}}}},
                                      {"required", json::array({"city"})}}}}}}});

    auto outcome = run(session, engine, request);
    TEST_REQUIRE(outcome.end == flm::request::ChatEnd::completed);
    TEST_REQUIRE(outcome.message.tool_calls.size() == 1);
    TEST_REQUIRE(outcome.message.tool_calls[0].name == "get_weather");
    const json expected_arguments = {{"city", "Paris"}};
    const json actual_arguments = json::parse(outcome.message.tool_calls[0].arguments);
    TEST_REQUIRE(actual_arguments == expected_arguments);
    TEST_REQUIRE(outcome.meta.stop_reason == TOOL_DETECTED);
}

void test_tool_call_does_not_promote_a_truncated_stop() {
    // Length, cancel and error win: a cut-off generation is never reported as a tool call.
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.use_format(chat_format_kind::HERMES_JSON, false);
    engine.reply =
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    engine.reply_stop = MAX_LENGTH_REACHED;

    auto request = make_request(json::array({message("user", "weather in Paris?")}));
    request.tools = json::array({json{
        {"type", "function"},
        {"function", {{"name", "get_weather"}, {"parameters", {{"type", "object"}}}}}}});

    auto outcome = run(session, engine, request);
    TEST_REQUIRE(outcome.message.tool_calls.size() == 1);
    TEST_REQUIRE(outcome.meta.stop_reason == MAX_LENGTH_REACHED);
}

void test_only_set_engine_parameters_reach_the_engine() {
    FakeEngine engine(model_directory());
    flm::request::ChatSession session;
    engine.reply = "ok";
    // Defaults of sampler_config: top_k 40, top_p 0.9, min_p 0.1, temperature 0.8,
    // presence 0, repetition 1, frequency 0.
    {
        auto request = make_request(json::array({message("user", "hi")}));
        request.parameters.temperature = 0.25f;
        run(session, engine, request);
        const Sampler& s = engine.sampler_state();
        TEST_REQUIRE(s.temperature == 0.25f);
        TEST_REQUIRE(s.top_k == 40);
        TEST_REQUIRE(s.top_p == 0.9f);
        TEST_REQUIRE(s.min_p == 0.1f);
        TEST_REQUIRE(s.pre_penalty == 0.0f);
        TEST_REQUIRE(s.rep_penalty == 1.0f);
        TEST_REQUIRE(s.freq_penalty == 0.0f);
        TEST_REQUIRE(engine.configured.empty());
    }
    {
        auto request = make_request(json::array({message("user", "hi")}));
        request.parameters.top_k = 5;
        request.parameters.top_p = 0.5f;
        request.parameters.min_p = 0.2f;
        request.parameters.presence_penalty = 0.3f;
        request.parameters.frequency_penalty = 0.4f;
        request.parameters.repetition_penalty = 1.5f;
        request.parameters.think = false;
        request.parameters.reasoning_effort = std::string("low");
        request.parameters.image_max_tokens = 256;
        run(session, engine, request);
        const Sampler& s = engine.sampler_state();
        TEST_REQUIRE(s.top_k == 5);
        TEST_REQUIRE(s.top_p == 0.5f);
        TEST_REQUIRE(s.min_p == 0.2f);
        TEST_REQUIRE(s.pre_penalty == 0.3f);
        TEST_REQUIRE(s.freq_penalty == 0.4f);
        TEST_REQUIRE(s.rep_penalty == 1.5f);
        TEST_REQUIRE(s.temperature == 0.25f);  // not in this request, so it keeps the earlier value

        TEST_REQUIRE(engine.configured.size() == 3);
        TEST_REQUIRE(engine.configured[0].first == "enable_think");
        TEST_REQUIRE(std::any_cast<bool>(engine.configured[0].second) == false);
        TEST_REQUIRE(engine.configured[1].first == "reasoning_effort");
        TEST_REQUIRE(std::any_cast<std::string>(engine.configured[1].second) == "low");
        TEST_REQUIRE(engine.configured[2].first == "image_max_tokens");
        TEST_REQUIRE(std::any_cast<int>(engine.configured[2].second) == 256);
    }
}

}  // namespace

int main() {
    RunTest(test_non_stream_completion_returns_parsed_message_and_meta,
            "non-stream completion returns the parsed message and meta");
    RunTest(test_stream_delivers_whole_utf8_characters,
            "stream delivers deltas and never splits a UTF-8 character");
    RunTest(test_stream_without_sink_is_rejected, "stream without a sink is rejected");
    RunTest(test_prompt_cache_reuses_an_extended_prompt,
            "prompt cache: extended prompt sets restore_allowed");
    RunTest(test_prompt_cache_misses_when_the_first_message_changed,
            "prompt cache: changed first message does not restore");
    RunTest(test_prompt_cache_is_not_used_when_the_request_opts_out,
            "prompt cache: allow_prompt_cache=false does not restore");
    RunTest(test_prompt_cache_reset_forces_a_miss,
            "prompt cache: reset forces a miss");
    RunTest(test_insert_refused_by_cancellation_is_prefill_cancelled,
            "insert refused while cancelled: prefill_cancelled and context cleared");
    RunTest(test_insert_refused_without_cancellation_is_context_exceeded,
            "insert refused without cancel: context_exceeded and context cleared");
    RunTest(test_runtime_error_from_generate_clears_and_rethrows,
            "generate throwing runtime_error clears the context and rethrows");
    RunTest(test_runtime_error_from_streamed_generate_clears_and_rethrows,
            "streamed generate throwing runtime_error clears the context and rethrows");
    RunTest(test_streamed_model_error_without_session_clear_leaves_context_to_engine,
            "streamed ModelRequestError(session_cleared=false) rethrown, context left to engine");
    RunTest(test_non_stream_model_error_clears_the_context_as_the_handler_did,
            "non-stream ModelRequestError clears the context (as the handler did)");
    RunTest(test_model_error_that_cleared_the_session_resets_the_cache,
            "ModelRequestError(session_cleared=true) resets the prompt cache");
    RunTest(test_model_error_without_session_clear_keeps_the_cache,
            "ModelRequestError(session_cleared=false) keeps the prompt cache");
    RunTest(test_single_turn_engine_rejects_history,
            "single-turn engine rejects an assistant message");
    RunTest(test_single_turn_engine_accepts_one_user_message,
            "single-turn engine accepts a single user message");
    RunTest(test_tool_call_in_output_promotes_the_stop_reason,
            "tool call in output promotes EOT_DETECTED to TOOL_DETECTED");
    RunTest(test_tool_call_does_not_promote_a_truncated_stop,
            "tool call does not promote a length stop");
    RunTest(test_only_set_engine_parameters_reach_the_engine,
            "only set engine parameters reach the engine");
    if (FailureCount() == 0) {
        std::cout << "All chat_session tests passed\n";
        return 0;
    }
    return 1;  // RunTest's atexit hook forces the exit code and prints the count
}
