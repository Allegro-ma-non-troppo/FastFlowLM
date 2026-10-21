/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file chat_request.cpp
 * \brief Transport-independent chat and completion request entry points
 * \note The helpers below and the two entry points moved here from RestHandler
 *       (src/server/rest_handler.cpp) without a change of behavior.
 */
#include "request/chat_request.hpp"

#include <iostream>
#include <memory>
#include <streambuf>
#include <utility>

#include "chat/chat_format.hpp"
#include "chat/chat_parser.hpp"
#include "chat/chat_stream.hpp"
#include "wstream_buf.hpp"

namespace flm::request {

namespace {

///@brief Normalize messages by merging consecutive user messages (like Ollama does)
///@param messages the original messages
///@return normalized messages with consecutive user messages merged
static json normalize_messages(json messages) {
    if (messages.empty()) return messages;

    json normalized = nlohmann::ordered_json::array();

    for (size_t i = 0; i < messages.size(); i++) {
        auto current_msg = messages[i];
        std::string role = current_msg.value("role", "");

        if (role == "user") {
            json merged_content_array = json::array();

            // Merge all consecutive user messages into array format
            while (i < messages.size() && messages[i].value("role", "") == "user") {
                if (messages[i].contains("content")) {
                    if (messages[i]["content"].is_array()) {
                        for (auto& item : messages[i]["content"]) {
                            merged_content_array.push_back(item);
                        }
                    }
                    else if (messages[i]["content"].is_string()) {
                        std::string text = messages[i]["content"].get<std::string>();
                        if (!text.empty()) {
                            nlohmann::ordered_json text_item;
                            text_item["type"] = "text";
                            text_item["text"] = text;
                            merged_content_array.push_back(text_item);
                        }
                    }
                }
                if (i + 1 < messages.size() && messages[i + 1].value("role", "") == "user") i++;
                else break;
            }

            current_msg["content"] = merged_content_array;
        }
        else if (role == "system") {
            json merged_content_array = json::array();

            // Merge all consecutive system messages into array format
            while (i < messages.size() && messages[i].value("role", "") == "system") {
                if (messages[i].contains("content")) {
                    if (messages[i]["content"].is_array()) {
                        for (auto& item : messages[i]["content"]) {
                            merged_content_array.push_back(item);
                        }
                    }
                    else if (messages[i]["content"].is_string()) {
                        std::string text = messages[i]["content"].get<std::string>();
                        if (!text.empty()) {
                            json text_item;
                            text_item["type"] = "text";
                            text_item["text"] = text;
                            merged_content_array.push_back(text_item);
                        }
                    }
                }
                if (i + 1 < messages.size() && messages[i + 1].value("role", "") == "system") i++;
                else break;
            }

            current_msg["content"] = merged_content_array;
        }
        else if (role == "assistant") {
            // Strip prior assistant "thinking" / reasoning fields so they are not
            // fed back into the model on subsequent turns.
            if (current_msg.contains("thinking")) {
                current_msg.erase("thinking");
            }
            if (current_msg.contains("reasoning")) {
                current_msg.erase("reasoning");
            }
            if (current_msg.contains("reasoning_content")) {
                current_msg.erase("reasoning_content");
            }
            // Also strip <think>...</think> blocks from string content if present.
            // if (current_msg.contains("content") && current_msg["content"].is_string()) {
            //     std::string text = current_msg["content"].get<std::string>();
            //     size_t start = text.find("<think>");
            //     while (start != std::string::npos) {
            //         size_t end = text.find("</think>", start);
            //         if (end == std::string::npos) {
            //             text.erase(start);
            //             break;
            //         }
            //         text.erase(start, (end + std::string("</think>").size()) - start);
            //         start = text.find("<think>");
            //     }
            //     // Trim leading whitespace/newlines left after removal.
            //     size_t first = text.find_first_not_of(" \t\r\n");
            //     if (first == std::string::npos) text.clear();
            //     else if (first > 0) text.erase(0, first);
            //     current_msg["content"] = text;
            // }
        }
        normalized.push_back(current_msg);
    }

    return normalized;
}

static json normalize_template(json messages) {
    json template_message = json::array();

    for (auto& message : messages) {
        json new_message = message;
        std::string merged_text;
        nlohmann::ordered_json::array_t merged_images;
        nlohmann::ordered_json::array_t merged_audio;

        if (message["content"].is_string()) {
            // Simple format: just text
            merged_text = message["content"].get<std::string>();
        }
        else if (message["content"].is_array()) {
            // Structured format: extract text and image URLs
            for (auto& contentItem : message["content"]) {
                if (contentItem.contains("type") && contentItem["type"] == "text") {
                    merged_text += contentItem["text"].get<std::string>();
                }
                else if (contentItem.contains("type") && contentItem["type"] == "image_url") {
                    std::string image_url = contentItem["image_url"]["url"].get<std::string>();
                    const std::vector<std::string> prefixes = {
                        "data:image/png;base64,",
                        "data:image/jpeg;base64,",
                        "data:image/jpg;base64,"
                    };
                    for (const auto& prefix : prefixes) {
                        if (image_url.substr(0, prefix.length()) == prefix) {
                            image_url = image_url.substr(prefix.length());
                            break;
                        }
                    }
                    if (image_url.substr(0, 5) == "data:") {
                        header_print("Warning", "Unsupported image format, skipping this image.");
                        continue;
                    }
                    if (image_url.empty()) {
                        header_print("Warning", "Empty image, skipping this image.");
                        continue;
                    }
                    merged_images.push_back(image_url);
                }
                else if (contentItem.contains("type") && contentItem["type"] == "input_audio") {
                    std::string audio_base64 = contentItem["input_audio"]["data"].get<std::string>();
                    if (audio_base64.empty()) {
                        header_print("Warning", "Empty audio, skipping this audio.");
                        continue;
                    }
                    merged_audio.push_back(audio_base64);
                }
            }
        }

        //new_message["role"] = message["role"];
        new_message["content"] = merged_text;
        if (!merged_images.empty()) {
            new_message["images"] = merged_images;
        }
        if (!merged_audio.empty()) {
            new_message["audios"] = merged_audio;
        }
        

        template_message.push_back(new_message);
    }

    return template_message;
}


///@brief Try to parse a JSON value from a string, returning the original string on failure.
static json try_parse_json_value(const std::string& s) {
    try {
        return json::parse(s);
    }
    catch (...) {
        return json(s);
    }
}

///@brief Convert OpenAI-style assistant tool_calls + following tool messages into the
/// Gemma4 chat-template format, which expects a single assistant message containing
/// both `tool_calls` and `tool_responses` (with `{name, response}` entries).
static json convert_tool_responses_gemma4(json messages) {
    json converted_messages = json::array();

    size_t i = 0;
    while (i < messages.size()) {
        const auto& msg = messages[i];

        if (msg.value("role", "") != "assistant" || !msg.contains("tool_calls") ||
            !msg.at("tool_calls").is_array() || msg.at("tool_calls").empty()) {
            converted_messages.push_back(msg);
            i++;
            continue;
        }

        // Normalize tool_calls: ensure function.arguments is a JSON object (not a string).
        json normalized_tool_calls = json::array();
        // Map tool_call_id -> function name, for matching tool responses below.
        std::unordered_map<std::string, std::string> id_to_name;
        // Preserve call order so unmatched tool responses can fall back positionally.
        std::vector<std::string> call_names_in_order;

        for (const auto& tc : msg.at("tool_calls")) {
            json new_tc = tc;
            std::string name;
            if (new_tc.contains("function") && new_tc["function"].is_object()) {
                auto& fn = new_tc["function"];
                if (fn.contains("name") && fn["name"].is_string()) {
                    name = fn["name"].get<std::string>();
                }
                if (fn.contains("arguments") && fn["arguments"].is_string()) {
                    fn["arguments"] = try_parse_json_value(fn["arguments"].get<std::string>());
                }
            }
            if (new_tc.contains("id") && new_tc["id"].is_string()) {
                id_to_name[new_tc["id"].get<std::string>()] = name;
            }
            call_names_in_order.push_back(name);
            normalized_tool_calls.push_back(new_tc);
        }

        // Collect consecutive following "tool" role messages as tool_responses.
        json tool_responses = json::array();
        size_t j = i + 1;
        size_t response_index = 0;
        while (j < messages.size() && messages[j].value("role", "") == "tool") {
            const auto& tool_msg = messages[j];

            // Resolve the function name: prefer matching by tool_call_id, then by
            // an explicit name field, then by positional order of the tool_calls.
            std::string name;
            if (tool_msg.contains("tool_call_id") && tool_msg["tool_call_id"].is_string()) {
                auto it = id_to_name.find(tool_msg["tool_call_id"].get<std::string>());
                if (it != id_to_name.end()) name = it->second;
            }
            if (name.empty() && tool_msg.contains("name") && tool_msg["name"].is_string()) {
                name = tool_msg["name"].get<std::string>();
            }
            if (name.empty() && response_index < call_names_in_order.size()) {
                name = call_names_in_order[response_index];
            }

            // Parse content into a JSON value if possible.
            json response_value;
            if (tool_msg.contains("content")) {
                const auto& content = tool_msg.at("content");
                if (content.is_string()) {
                    response_value = try_parse_json_value(content.get<std::string>());
                }
                else {
                    response_value = content;
                }
            }
            else {
                response_value = nullptr;
            }

            tool_responses.push_back({
                {"name", name},
                {"response", response_value},
            });

            response_index++;
            j++;
        }

        json merged = {
            {"role", "assistant"},
            {"tool_calls", normalized_tool_calls},
        };
        if (!tool_responses.empty()) {
            merged["tool_responses"] = tool_responses;
        }
        if (msg.contains("content") && !msg.at("content").is_null()) {
            merged["content"] = msg.at("content");
        }
        if (msg.contains("reasoning_content") && !msg.at("reasoning_content").is_null()) {
            merged["reasoning_content"] = msg.at("reasoning_content");
        }

        converted_messages.push_back(merged);
        i = j;
    }

    return converted_messages;
}

/// \brief Parses the generated text as it is written and hands the changes to a sink
/// \note The text arrives in pieces, possibly in the middle of a UTF-8 sequence. A piece is
///       parsed on every flush, and only the complete sequences of the buffer are parsed.
///       finish() parses the rest, incomplete sequences included, and lets the parser release
///       whatever it held back.
class DeltaStreambuf : public std::streambuf {
public:
    DeltaStreambuf(const chat_format_spec& spec, std::string prefill, const chat_parse_options& options, ChatSink& sink)
        : stream_(spec, std::move(prefill), options), sink_(sink) {}

    void finish() {
        send(buffer_, true);
        buffer_.clear();
    }

    const chat_msg& message() const { return stream_.message(); }

protected:
    int_type overflow(int_type ch) override {
        if (ch != traits_type::eof()) {
            buffer_ += static_cast<char>(ch);
        }
        return ch;
    }

    int sync() override {
        flush_complete_utf8_sequences();
        return 0;
    }

private:
    static size_t utf8_sequence_length(unsigned char first_byte) {
        if ((first_byte & 0x80) == 0) {
            return 1;
        }
        if ((first_byte & 0xE0) == 0xC0) {
            return 2;
        }
        if ((first_byte & 0xF0) == 0xE0) {
            return 3;
        }
        if ((first_byte & 0xF8) == 0xF0) {
            return 4;
        }
        return 0; // invalid start byte
    }

    void flush_complete_utf8_sequences() {
        if (buffer_.empty()) {
            return;
        }
        std::string complete;
        size_t pos = 0;
        while (pos < buffer_.size()) {
            const size_t length = utf8_sequence_length(static_cast<unsigned char>(buffer_[pos]));
            if (length == 0) {
                pos++; // skip an invalid start byte
                continue;
            }
            if (pos + length > buffer_.size()) {
                break; // incomplete sequence, wait for the rest
            }
            complete.append(buffer_, pos, length);
            pos += length;
        }
        if (!complete.empty()) {
            send(complete, false);
        }
        if (pos > 0) {
            buffer_.erase(0, pos);
        }
    }

    void send(const std::string& text, bool generation_done) {
        for (const chat_msg_delta& delta : stream_.update(text, generation_done)) {
            sink_.on_delta(delta);
        }
    }

    std::string buffer_;
    chat_stream stream_;
    ChatSink& sink_;
};

class DeltaStream : public std::ostream {
public:
    DeltaStream(const chat_format_spec& spec, std::string prefill, const chat_parse_options& options, ChatSink& sink)
        : std::ostream(&buf_), buf_(spec, std::move(prefill), options, sink) {}
    void finish() { buf_.finish(); }
    const chat_msg& message() const { return buf_.message(); }

private:
    DeltaStreambuf buf_;
};

constexpr int kDefaultLengthLimit = 4096;

/// \brief Only a clean stop can become a tool-call stop; length, cancel and error win, so a
///        truncated or aborted generation is never reported as a complete tool call.
void promote_tool_stop_reason(chat_meta_info_t& meta, const chat_msg& message) {
    if (!message.tool_calls.empty() && meta.stop_reason == stop_reason_t::EOT_DETECTED) {
        meta.stop_reason = stop_reason_t::TOOL_DETECTED;
    }
}

} // namespace

void apply_engine_parameters(AutoModel& engine, const EngineParameters& p) {
    if (p.temperature) {
        engine.set_temperature(*p.temperature);
    }
    if (p.top_p) {
        engine.set_topp(*p.top_p);
    }
    if (p.top_k) {
        engine.set_topk(*p.top_k);
    }
    if (p.min_p) {
        engine.set_minp(*p.min_p);
    }
    if (p.presence_penalty) {
        engine.set_presence_penalty(*p.presence_penalty);
    }
    if (p.frequency_penalty) {
        engine.set_frequency_penalty(*p.frequency_penalty);
    }
    if (p.repetition_penalty) {
        engine.set_repetition_penalty(*p.repetition_penalty);
    }
    if (p.think) {
        engine.configure_parameter("enable_think", *p.think);
    }
    if (p.reasoning_effort) {
        engine.configure_parameter("reasoning_effort", *p.reasoning_effort);
    }
    if (p.image_max_tokens) {
        engine.configure_parameter("image_max_tokens", *p.image_max_tokens);
    }
}

void ChatSession::reset_prompt_cache() {
    prompt_cache_.reset();
}

ChatOutcome ChatSession::run_chat(AutoModel& engine, const ChatRequest& request, ChatSink* sink,
                                  const std::function<bool()>& is_cancelled) {
    if (request.stream && sink == nullptr) {
        throw std::invalid_argument("a streaming chat request needs a sink");
    }

    ChatOutcome outcome;
    chat_meta_info_t& meta_info = outcome.meta;

    // Everything up to the first engine call counts as setup: a failure there is the request's
    // own, not the engine's, and a transport reports the two differently.
    json messages;
    json tools;
    lm_uniform_input_t uniformed_input;
    int length_limit = kDefaultLengthLimit;
    chat_parse_options parse_options;
    try {
        apply_engine_parameters(engine, request.parameters);

        messages = normalize_messages(request.messages);
        messages = normalize_template(messages);
        tools = request.tools;

        // see if we can use the prompt cache
        if (engine.single_turn) {
            // reject multi-turn requests: any assistant message in the history means the caller is
            // trying to continue a prior turn, which this model cannot support.
            for (const auto& msg : messages) {
                if (msg.value("role", "") == "assistant") {
                    header_print("Warning", "Single-turn model received multi-turn request (assistant messages detected). Rejecting.");
                    outcome.end = ChatEnd::single_turn_rejected;
                    return outcome;
                }
            }
        }
        else {
            // A model switch needs no case of its own: the host resets the cache whenever it
            // replaces the engine, so the first request on a new engine is a miss.
            cache_match_info_t cache_info;
            const bool cache_matches =
                prompt_cache_.can_use_cache(messages, engine.get_chat_template_type(), tools, cache_info);
            if (cache_matches && request.allow_prompt_cache) {
                meta_info.restore_allowed = true;
                header_print("FLM", "Use cached prompt!");
                size_t matched_rounds = cache_info.matched_rounds + (engine.check_using_checkpint() ? 0 : 1);
                header_print("FLM", "Matched " + std::to_string(matched_rounds) +
                    " out of " + std::to_string(cache_info.total_rounds) + " messages (" +
                    std::to_string(cache_info.total_rounds - matched_rounds) + " new to prefill).");
            }
            else {
                // cannot use the cache (or the request opted out), clear and re-insert all
                header_print("FLM", cache_matches ? "Prompt cache disabled for this request."
                                                  : "Prompt cache miss.");
                header_print("FLM", "Clearing context...");
                engine.clear_context();
            }
        }

        if (request.model.starts_with("gemma4-it")) {
            messages = convert_tool_responses_gemma4(messages);
        }

        uniformed_input.messages = messages;
        uniformed_input.tools = tools;
        uniformed_input.requested_max_new_tokens = normalize_requested_max_new_tokens(request.max_new_tokens);
        length_limit = request.max_new_tokens.value_or(kDefaultLengthLimit);
        meta_info.load_duration = request.load_duration_ns;
        meta_info.max_prefill_len = request.max_prefill_len;

        chat_constraint_request constraint = request.constraint;
        constraint.tools = &tools;
        try {
            meta_info.constraint = engine.make_constraint(constraint);
        }
        catch (const std::invalid_argument& e) {
            // the request asks for something this model's format cannot do, or a bad schema
            throw InvalidRequest(e.what());
        }

        // tool markup is only a tool call when the request offered tools
        parse_options.tools = &tools;
        parse_options.parse_tool_calls = !tools.empty() && constraint.tool_choice != chat_tool_choice::NONE;
    }
    catch (const InvalidRequest&) {
        throw;
    }
    catch (const ModelRequestError&) {
        throw;
    }
    catch (const std::exception& e) {
        throw RequestSetupError(e.what());
    }

    // what a request used is read before a failure or a cancellation clears the context
    auto snapshot_kv = [&] {
        outcome.active_kv_tokens = engine.get_current_context_length();
        outcome.max_kv_tokens = engine.get_max_length();
    };
    auto drop_context = [&] {
        engine.clear_context();
        prompt_cache_.reset();
    };

    header_print("FLM", "Start prefill...");
    try {
        const bool inserted = engine.insert(meta_info, uniformed_input, is_cancelled);
        if (!inserted) {
            snapshot_kv();
            if (meta_info.stop_reason == CANCEL_DETECTED || is_cancelled()) {
                meta_info.stop_reason = CANCEL_DETECTED;
                header_print("❌ ", "Prefill Cancelled!");
                outcome.end = ChatEnd::prefill_cancelled;
            }
            else {
                outcome.end = ChatEnd::context_exceeded;
            }
            drop_context();
            return outcome;
        }
    }
    catch (const std::exception&) {
        drop_context();
        throw;
    }

    const chat_format_spec& spec = engine.get_chat_format();
    const std::string prefill = chat_format_prefill(spec, meta_info.prompt_tail);

    header_print("FLM", "Start generating...");
    if (request.stream) {
        DeltaStream ostream(spec, prefill, parse_options, *sink);
        try {
            engine.generate(meta_info, length_limit, ostream, is_cancelled);
        }
        catch (const ModelRequestError& error) {
            // the engine says whether it cleaned its session itself
            if (error.session_cleared()) {
                prompt_cache_.reset();
            }
            throw;
        }
        catch (const std::exception&) {
            drop_context();
            throw;
        }
        if (meta_info.stop_reason == CANCEL_DETECTED) {
            header_print("❌ ", "Generation Cancelled!");
            prompt_cache_.reset();
            outcome.end = ChatEnd::cancelled;
        }
        // send everything left, including incomplete sequences
        ostream.finish();
        outcome.message = ostream.message();
    }
    else {
        nullstream discard;
        std::string text;
        try {
            text = engine.generate(meta_info, length_limit, discard, is_cancelled);
        }
        catch (const std::exception&) {
            drop_context();
            throw;
        }
        outcome.message = chat_parse(spec, prefill + text, false, parse_options);
        if (meta_info.stop_reason == CANCEL_DETECTED) {
            header_print("❌ ", "Generation Cancelled!");
            prompt_cache_.reset();
            outcome.end = ChatEnd::cancelled;
        }
    }
    promote_tool_stop_reason(meta_info, outcome.message);
    snapshot_kv();
    return outcome;
}

CompletionOutcome ChatSession::run_completion(AutoModel& engine, const CompletionRequest& request, std::ostream& os,
                                              const std::function<bool()>& is_cancelled) {
    CompletionOutcome outcome;
    chat_meta_info_t& meta_info = outcome.meta;

    // The engine is shared with chat requests and this one replaces its conversation, so the chat
    // cache no longer describes it.
    prompt_cache_.reset();
    apply_engine_parameters(engine, request.parameters);

    meta_info.max_prefill_len = request.max_prefill_len;
    lm_uniform_input_t uniformed_input;
    uniformed_input.requested_max_new_tokens = normalize_requested_max_new_tokens(request.max_new_tokens);
    uniformed_input.prompt = request.prompt;
    const int length_limit = request.max_new_tokens.value_or(kDefaultLengthLimit);

    header_print("FLM", "Start generating...");
    try {
        if (!engine.insert(meta_info, uniformed_input, is_cancelled)) {
            engine.clear_context();
            outcome.context_exceeded = true;
            return outcome;
        }
    }
    catch (const std::exception&) {
        engine.clear_context();
        throw;
    }
    try {
        engine.generate(meta_info, length_limit, os, is_cancelled);
    }
    catch (const ModelRequestError&) {
        // a streamed completion leaves the context to the engine, as it always did
        if (!request.stream) {
            engine.clear_context();
        }
        throw;
    }
    catch (const std::exception&) {
        engine.clear_context();
        throw;
    }
    if (request.stream) {
        engine.clear_context();
    }
    return outcome;
}

} // namespace flm::request
