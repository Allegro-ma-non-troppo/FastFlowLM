/// \file chat_request.hpp
/// \brief Transport-independent entry points for chat and completion requests
/// \note The request-level logic of /v1/chat/completions and /v1/completions used to be
///       written inside the HTTP handler: message normalization, the prompt cache
///       decision, the single-turn check, insert and generate, cancellation cleanup and
///       output parsing. It lives here so that RestHandler and any program that links
///       FastFlowLM run the same code. Nothing in this header knows about HTTP, JSON
///       envelopes or server-sent events.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "AutoModel/automodel.hpp"
#include "chat/chat_constraint.hpp"
#include "chat/chat_msg.hpp"
#include "prompt_cache.hpp"

namespace flm::request {

/// \brief Sampling and template parameters of a request
/// \note Each field is set only when the request carried it. The engine keeps the value
///       of a field the request left out, so omitting a field is not the same as
///       sending its default.
struct EngineParameters {
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<int> top_k;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<float> repetition_penalty;
    std::optional<bool> think;                   // configure_parameter("enable_think")
    std::optional<std::string> reasoning_effort; // configure_parameter("reasoning_effort")
    std::optional<int> image_max_tokens;         // configure_parameter("image_max_tokens")
};

/// \brief Apply the fields that are set to the engine
void apply_engine_parameters(AutoModel& engine, const EngineParameters& parameters);

/// \brief A chat request, after the transport has parsed it
struct ChatRequest {
    /// \brief the model tag; tags starting with gemma4-it get the Gemma 4 tool-response conversion
    std::string model;
    /// \brief OpenAI messages exactly as received, before normalization
    nlohmann::ordered_json messages;
    /// \brief the tools offered to the model; an empty array when there are none or tool_choice is "none"
    nlohmann::ordered_json tools = nlohmann::ordered_json::array();
    /// \brief the decoding constraint; its tools pointer is set by run_chat
    chat_constraint_request constraint;
    /// \brief the output budget the caller asked for; 4096 tokens when unset
    std::optional<int> max_new_tokens;
    EngineParameters parameters;
    /// \brief false makes a prompt cache hit count as a miss: the context is cleared and the whole prompt is prefilled again
    bool allow_prompt_cache = true;
    /// \brief the prefill chunk length of the loaded model, copied to the request's meta info
    int max_prefill_len = 0;
    /// \brief how long loading the model took for this request, reported in the usage
    uint64_t load_duration_ns = 0;
    /// \brief true: deltas go to the sink while generating; false: the whole message is returned in the outcome
    bool stream = false;
};

/// \brief A completion request (a raw prompt, no messages)
struct CompletionRequest {
    std::string prompt;
    std::optional<int> max_new_tokens;
    EngineParameters parameters;
    int max_prefill_len = 0;
    /// \brief the caller streams the output; the KV is dropped after a successful streamed completion,
    ///        and a ModelRequestError from generate does not clear the context again
    bool stream = false;
};

/// \brief The request asks for something the model cannot do (a constraint that cannot be built)
class InvalidRequest : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

/// \brief Preparing the request failed before the engine was touched (messages that cannot be normalized, ...)
/// \note Kept apart from the engine's own exceptions because a transport reports the two differently.
class RequestSetupError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// \brief Receives the parts of the assistant message while it is generated (stream == true)
class ChatSink {
public:
    virtual ~ChatSink() = default;
    /// \brief one change of the assistant message, in order
    virtual void on_delta(const chat_msg_delta& delta) = 0;
};

/// \brief How a request ended when no exception was thrown
enum class ChatEnd {
    completed,           // generated until a stop condition other than cancellation
    cancelled,           // cancelled while generating; the message holds what was produced
    prefill_cancelled,   // cancelled while the prompt was inserted; nothing was generated
    context_exceeded,    // the prompt does not fit the context; the context was cleared
    single_turn_rejected // the model takes single-turn requests and the history has an assistant message
};

/// \brief Result of run_chat
struct ChatOutcome {
    ChatEnd end = ChatEnd::completed;
    /// \brief usage, timings and stop reason; a parsed tool call has promoted EOT_DETECTED to TOOL_DETECTED
    chat_meta_info_t meta;
    /// \brief the assistant message parsed from the output; set for completed and cancelled
    chat_msg message;
    /// \brief tokens in the context and the context capacity, read before any cleanup so that a
    ///        cancelled request reports what it used
    int active_kv_tokens = 0;
    unsigned int max_kv_tokens = 0;
};

/// \brief Result of run_completion
struct CompletionOutcome {
    bool context_exceeded = false;
    chat_meta_info_t meta;
};

/// \brief The state that spans requests: the prompt cache and the model of the previous request
/// \note One session per engine owner. Calls must not overlap; the callers already serialize
///       NPU access.
class ChatSession {
public:
    /// \brief Run one chat request on an engine whose model is loaded
    /// \param engine the loaded model
    /// \param request the parsed request
    /// \param sink receives the deltas when request.stream is set; may be null otherwise
    /// \param is_cancelled polled during prefill and decoding
    /// \return how the request ended
    /// \throws InvalidRequest when the constraint the request asks for cannot be built
    /// \throws RequestSetupError when the request cannot be prepared for the engine
    /// \throws ModelRequestError, std::exception from insert or generate, after the context and the
    ///         prompt cache have been cleaned up the same way the HTTP handler always did
    ChatOutcome run_chat(AutoModel& engine, const ChatRequest& request, ChatSink* sink,
                         const std::function<bool()>& is_cancelled);

    /// \brief Run one completion request on an engine whose model is loaded
    /// \note The completion replaces the conversation held by the engine, so the prompt cache is reset first
    /// \param os receives the generated text
    /// \throws ModelRequestError, std::exception from insert or generate, after the context was cleared
    CompletionOutcome run_completion(AutoModel& engine, const CompletionRequest& request, std::ostream& os,
                                     const std::function<bool()>& is_cancelled);

    /// \brief Forget the cached prompt, so the next request misses
    void reset_prompt_cache();

private:
    PromptCache prompt_cache_;
};

} // namespace flm::request
