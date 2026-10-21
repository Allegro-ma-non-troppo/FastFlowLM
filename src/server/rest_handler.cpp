/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file rest_handler.cpp
 * \brief RestHandler class and related declarations
 * \author FastFlowLM Team
 * \date 2025-08-05
 *  \version 0.9.24
 */
#include "rest_handler.hpp"
#include "wstream_buf.hpp"
#include "streaming_ostream.hpp"
#include "streaming_ostream_openai.hpp"
#include "image/image_reader.hpp"
#include "whisper/transcript_format.hpp"
#include <sstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <locale>
#include <random>
#include "server.hpp"

namespace {
json ModelErrorResponse(const ModelRequestError& error) {
    return {{"error", {{"message", error.what()},
                       {"type", "model_error"},
                       {"code", error.http_code()},
                       {"session_cleared", error.session_cleared()}}}};
}

json ExceptionResponse(const std::exception& error) {
    if (const auto* model_error = dynamic_cast<const ModelRequestError*>(&error))
        return ModelErrorResponse(*model_error);
    return {{"error", error.what()}};
}

/// \brief Whether a request sets a field
/// \note Some clients send null for every option they leave unset, so null
///       counts as not set; json::value() and conversions throw on it instead.
bool has_value(const json& request, const char* key) {
    auto it = request.find(key);
    return it != request.end() && !it->is_null();
}

/// \brief A request field, or the fallback when it is missing or null
template <typename T>
T value_or(const json& request, const char* key, T fallback) {
    return has_value(request, key) ? request.at(key).get<T>() : fallback;
}
}

///@brief Whether a streaming OpenAI request asked for usage through stream_options.include_usage.
/// Following OpenAI, a stream reports no usage unless asked; a malformed option counts as not asked.
static bool stream_include_usage(const json& request) {
    auto options = request.find("stream_options");
    if (options == request.end() || !options->is_object()) {
        return false;
    }
    auto include_usage = options->find("include_usage");
    return include_usage != options->end() && include_usage->is_boolean() && include_usage->get<bool>();
}

///@brief RestHandler constructor
///@param models the model list
///@param downloader the downloader
///@param default_tag the default tag
///@param asr whether to enable asr
///@param embed whether to enable embedding

///@return the rest handler
RestHandler::RestHandler(model_list& models, ModelDownloader& downloader, program_args_t& args,
                         flm_rt::device* npu_device)
    : supported_models(models), downloader(downloader), default_model_tag(args.model_tag), current_model_tag(""), modelscope(args.modelscope), asr(args.asr), embed(args.embed), img_pre_resize(args.img_pre_resize), preemption(args.preemption), backend(args.backend), npu_device_inst(npu_device){

    if (args.ctx_length != -1) {
        this->ctx_length = args.ctx_length >= 512 ? args.ctx_length : 512;
    } else {
        this->ctx_length = -1;
    }
    if (args.prefill_chunk_len != -1) {
        this->prefill_chunk_len_arg = args.prefill_chunk_len >= 512 ? args.prefill_chunk_len : 512;
    }
    else {
        this->prefill_chunk_len_arg = -1;
    }
    this->prefill_chunk_len = this->prefill_chunk_len_arg;
    
    // Initialize chat bot with default model
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
    if (this->asr) {
        std::string whisper_tag = "whisper-v3:turbo";
        ensure_asr_model_loaded(whisper_tag);
    }
    if (this->embed) {
        std::string embed_tag = "embed-gemma:300m";
        ensure_embed_model_loaded(embed_tag);
    }
#else
    if (this->asr) {
        header_print("Error", "ASR models are not supported in this build");
    }
    if (this->embed) {
        header_print("Error", "Embedding models are not supported in this build");
    }
#endif

    if (default_model_tag != "model-faker") {
        if (!supported_models.is_model_supported(default_model_tag)) {
            header_print("Warning", "Default model tag '" << default_model_tag << "' is not supported. Falling back to 'llama3.2:1b'.");
            this->default_model_tag = "llama3.2:1b";
        }
        if (!ensure_model_loaded(default_model_tag)) {
            header_print("Error", "Failed to load default model: " + default_model_tag);
        }
    }
    else {
        this->current_model_tag = "model-faker";
    }
    chat_session.reset_prompt_cache();
}

///@brief RestHandler destructor
///@return the rest handler
RestHandler::~RestHandler() = default;

///@brief Ensure the model is loaded
///@param model_tag the model tag
bool RestHandler::ensure_model_loaded(const std::string& model_tag,
                                      const std::string& request_backend) {
    // Refuse a name the catalog does not have before anything resolves it;
    // an unknown name used to load a substitute model (and on dev04 could
    // end the process). Checked after rectify_model_tag, as get_model_info
    // does, so "Ollama/<tag>" and bare family names still resolve.
    // "model-faker" means no model was chosen.
    if (model_tag != "model-faker" &&
        !supported_models.is_model_supported(
            supported_models.rectify_model_tag(model_tag))) {
        header_print_r("ERROR", "Model tag '" << model_tag
            << "' is not supported. Please check with `flm list`.");
        return false;
    }

    // Nothing was chosen, so there is nothing to load; succeed only if a
    // model is in fact loaded.
    if (model_tag == "model-faker") {
        return auto_chat_engine != nullptr;
    }

    // A per-request "backend" overrides --backend; either one differing from
    // what is loaded forces a reload, exactly as a different model tag does.
    const std::string requested_backend =
        request_backend.empty() ? this->backend : request_backend;
    // Compared in its resolved form, which is what current_model_tag holds, so
    // "llama3.2" or "Ollama/llama3.2:1b" does not reload llama3.2:1b each time.
    const std::string resolved_tag = supported_models.rectify_model_tag(model_tag);
    if (current_model_tag != resolved_tag || current_backend != requested_backend) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        // The old model is released before the new one is built, so from here
        // on nothing is loaded until the load below succeeds. Record that now:
        // a failure must not leave the old tag naming an engine that is gone,
        // or the next request for it would skip the reload and use that engine.
        // The device is owned by main() and shared with every engine, so it is
        // deliberately left alone; releasing the engine is what frees the
        // hardware context.
        this->auto_chat_engine.reset();
        this->current_model_tag = "model-faker";
        this->current_backend.clear();
        // The cached conversation lived in the engine just released.
        this->chat_session.reset_prompt_cache();
        try {
            auto [ensure_tag, engine] = get_auto_model(model_tag, this->supported_models, this->npu_device_inst);
            switch (downloader.is_model_downloaded(ensure_tag, false, /*fast_check=*/true)) {
                case ModelDownloader::ModelStatus::Ready:
                    break;
                case ModelDownloader::ModelStatus::Outdated:
                case ModelDownloader::ModelStatus::Missing:
                    if (!downloader.pull_model(ensure_tag, this->modelscope)) {
                        throw std::runtime_error("failed to download " + ensure_tag);
                    }
                    break;
                case ModelDownloader::ModelStatus::Incompatible:
                    throw std::runtime_error(ensure_tag + " is incompatible with this version of FastFlowLM");
            }
            auto [new_ensure_tag, model_info] = supported_models.get_model_info(ensure_tag);
            engine->set_server_mode(true);
            engine->configure_parameter("img_pre_resize", this->img_pre_resize);
            const auto load_started = std::chrono::steady_clock::now();
            engine->load_model(supported_models.get_model_path(new_ensure_tag), model_info, ctx_length, preemption, requested_backend);
            report_load_time(load_started);

            // Per model, not per process: the flash models take 1024-token
            // chunks and the rest 4096, so keeping the first model's value
            // feeds every later model chunks sized for another one.
            this->prefill_chunk_len = this->prefill_chunk_len_arg != -1
                ? this->prefill_chunk_len_arg
                : model_info["max_prefill_len"].get<int>();
            this->auto_chat_engine = std::move(engine);
            this->current_model_tag = ensure_tag;
            this->current_backend = requested_backend;
        }
        catch (const std::exception& e) {
            header_print("ERROR", "Failed to load model: " + std::string(e.what()));
            return false;
        }
    }
    return auto_chat_engine != nullptr;
}

///@brief Ensure the asr model is loaded
///@param model_tag the model tag
void RestHandler::ensure_asr_model_loaded(const std::string& model_tag) {
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
    std::string ensure_tag = model_tag;
    switch (downloader.is_model_downloaded(ensure_tag, false, /*fast_check=*/true)) {
        case ModelDownloader::ModelStatus::Ready:
            break;
        case ModelDownloader::ModelStatus::Outdated:
        case ModelDownloader::ModelStatus::Missing:
            downloader.pull_model(ensure_tag, modelscope);
            break;
        case ModelDownloader::ModelStatus::Incompatible:
            header_print("ERROR", "Whisper is incompatible with this version of FastFlowLM, skipping... ");
            this->asr = false;
            return;
    }
    this->whisper_engine = std::make_unique<Whisper>(this->npu_device_inst);
    auto [new_ensure_tag, whisper_model_info] = this->supported_models.get_model_info(ensure_tag);
    std::string whisper_model_path = this->supported_models.get_model_path(new_ensure_tag);
    try {
        this->whisper_engine->load_model(whisper_model_path, whisper_model_info, this->preemption);
    }
    catch (const std::exception& e) {
        header_print("ERROR", "Failed to load ASR model: " + std::string(e.what()));
        exit(EXIT_FAILURE);
    }
#else
    throw std::runtime_error("ASR models are not supported in this build");
#endif
}

///@brief Ensure the embed model is loaded
///@param model_tag the model tag
void RestHandler::ensure_embed_model_loaded(const std::string& model_tag) {
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
    std::string ensure_tag = model_tag;
    switch (this->downloader.is_model_downloaded(ensure_tag, false, /*fast_check=*/true)) {
        case ModelDownloader::ModelStatus::Ready:
            break;
        case ModelDownloader::ModelStatus::Outdated:
        case ModelDownloader::ModelStatus::Missing:
            this->downloader.pull_model(ensure_tag, this->modelscope);
            break;
        case ModelDownloader::ModelStatus::Incompatible:
            header_print("ERROR", "EmbeddingGemma is incompatible with this version of FastFlowLM, skipping... ");
            this->embed = false;
            return;
    }
    auto [embedding_model_tag, auto_embedding_engine] = get_auto_embedding_model(ensure_tag, this->npu_device_inst);
    this->auto_embedding_engine = std::move(auto_embedding_engine);
    auto [new_embedding_model_tag, embedding_model_info] = this->supported_models.get_model_info(embedding_model_tag);
    std::string embedding_model_path = this->supported_models.get_model_path(new_embedding_model_tag);
    try {
        this->auto_embedding_engine->load_model(embedding_model_path, embedding_model_info, this->preemption);
    }
    catch (const std::exception& e) {
        header_print("ERROR", "Failed to load embedding model: " + std::string(e.what()));
        exit(EXIT_FAILURE);
    }
#else
    throw std::runtime_error("Embedding models are not supported in this build");
#endif
}

///@brief Read the sampling and template parameters a request carries
///@param request the request JSON object
///@note FastFlowLM reads min_p as an integer here, so a fractional min_p is rejected or truncated.
///      This is the behavior of the handler this code was moved from and is kept as is.
static flm::request::EngineParameters parse_engine_parameters(const json& request) {
    flm::request::EngineParameters parameters;
    if (has_value(request, "temperature")) {
        parameters.temperature = request["temperature"].get<float>();
    }
    if (has_value(request, "top_p")) {
        parameters.top_p = request["top_p"].get<float>();
    }
    if (has_value(request, "top_k")) {
        parameters.top_k = request["top_k"].get<int>();
    }
    if (has_value(request, "min_p")) {
        parameters.min_p = static_cast<float>(request["min_p"].get<int>());
    }
    if (has_value(request, "presence_penalty")) {
        parameters.presence_penalty = request["presence_penalty"].get<float>();
    }
    if (has_value(request, "frequency_penalty")) {
        parameters.frequency_penalty = request["frequency_penalty"].get<float>();
    }
    if (has_value(request, "repetition_penalty")) {
        parameters.repetition_penalty = request["repetition_penalty"].get<float>();
    }
    if (has_value(request, "think")) {
        parameters.think = request["think"].get<bool>();
    }
    if (has_value(request, "reasoning_effort")) {
        parameters.reasoning_effort = request["reasoning_effort"].get<std::string>();
    }
    if (has_value(request, "image-max-tokens")) {
        parameters.image_max_tokens = request["image-max-tokens"].get<int>();
    }
    return parameters;
}

///@brief The output budget of an OpenAI chat request: max_tokens, else max_completion_tokens
static std::optional<int> openai_chat_budget(const json& request) {
    return has_value(request, "max_tokens")
        ? std::optional<int>(request.at("max_tokens").get<int>())
        : has_value(request, "max_completion_tokens")
            ? std::optional<int>(request.at("max_completion_tokens").get<int>())
            : std::nullopt;
}

///@brief The choices array of a non-streaming chat completion
///@param parsed the assistant message parsed from the output
///@param meta_info the request's meta info, its stop reason is already final
static json build_chat_choices(const chat_msg& parsed, const chat_meta_info_t& meta_info) {
    json message;
    message["role"] = "assistant";
    if (!parsed.reasoning.empty()) {
        message["reasoning_content"] = parsed.reasoning;
    }

    if (!parsed.tool_calls.empty()) {
        json tool_calls_json = json::array();
        for (size_t i = 0; i < parsed.tool_calls.size(); i++) {
            tool_calls_json.push_back({
                {"index", i},
                {"id", chat_new_tool_call_id()},
                {"type", "function"},
                {"function", {
                    {"name", parsed.tool_calls[i].name},
                    {"arguments", parsed.tool_calls[i].arguments}
                }}
            });
        }
        message["tool_calls"] = tool_calls_json;
        // OpenAI sends null rather than "" when the turn is only tool calls
        message["content"] = parsed.content.empty() ? json(nullptr) : json(parsed.content);
    }
    else {
        message["content"] = parsed.content;
    }

    return json::array({
        {
            {"index", 0},
            {"message", message},
            {"logprobs", nullptr},
            {"finish_reason", stop_reason_to_string(meta_info.stop_reason)}
        }
    });
}

///@brief Configure chat engine parameters from options and request
///@param options the options JSON object
///@param request the request JSON object
void RestHandler::configure_chat_engine_parameters(const json& options, const json& request) {
    flm::request::apply_engine_parameters(*auto_chat_engine, parse_engine_parameters(request));
}
///@brief Handle the show request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_show(const json& request,
    std::function<void(const json&)> send_response,
    StreamResponseCallback send_streaming_response) {
    try {
        std::string model = request["model"];
        json info = {
            {"modelfile", ""},
            {"parameters", ""},
            {"template", ""},
            {"details", {
                {"parent_model", ""},
                {"format", ""},
                {"family", ""},
                {"families", {""}},
                {"parameter_size", ""},
                {"quantization_level", ""}
                }
            },
            {"model_info", {
                {"general.architecture", "flm" }
            }},
            {"capabilities", {"chat", "vision", "completion"}}
        };


        send_response(info);
    }
    catch (const std::exception& e) {
        json error_response = { {"error", e.what()} };
        send_response(error_response);
    }
}

///@brief Handle the generate request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_generate(const json& request,
                                 std::function<void(const json&)> send_response,
                                 StreamResponseCallback send_streaming_response,
                                 std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        std::string prompt = request["prompt"];
        bool stream = request.value("stream", true);
        std::string model = request.value("model", current_model_tag);
        const std::string request_backend = request.value("backend", std::string());
        json options = request.value("options", json::object());
       
        int length_limit = request.value("max_tokens", 4096);
        auto load_start_time = time_utils::now();
        // TODO: Use Another Check Function avoid loading again
        if (!ensure_model_loaded(model, request_backend)) {
            json error_response = {{"error", "Failed to load " + model + " model!"}};
            send_response(error_response);
            return;
        }
        auto load_end_time = time_utils::now();
      
        chat_meta_info_t meta_info;
        lm_uniform_input_t uniformed_input;
        uniformed_input.requested_max_new_tokens =
            normalize_requested_max_new_tokens(
                request.contains("max_tokens")
                    ? std::optional<int>(request.at("max_tokens").get<int>())
                    : std::nullopt);
        meta_info.max_prefill_len = this->prefill_chunk_len;
        meta_info.load_duration = (uint64_t)time_utils::duration_ns(load_start_time, load_end_time).first;
        header_print("FLM", "Start generating...");
        
        if (stream) {
            // Streaming response using streaming_ostream
            auto total_start_time = time_utils::now();
            streaming_ostream ostream(model, send_streaming_response, false);
            uniformed_input.prompt = prompt;
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input, [&] { return cancellation_token && cancellation_token->cancelled(); });
                if (!success){
                    json error_response = {{"error", "Max length reached"}};
                    send_response(error_response);
                    this->auto_chat_engine->clear_context();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            try {
                auto_chat_engine->generate(meta_info, length_limit, ostream, [&] { return cancellation_token && cancellation_token->cancelled(); });
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            auto total_end_time = time_utils::now();
            auto history = this->auto_chat_engine->get_history();
            // std::cout << "history: " << history.first << std::endl;
            meta_info.total_duration = (uint64_t)time_utils::duration_ns(total_start_time, total_end_time).first;
            ostream.finalize_generate(meta_info, history.second);
        } else {
            // Non-streaming response
            std::stringstream ss;
            wstream_buf obuf(ss);
            std::ostream ostream(&obuf);
            uniformed_input.prompt = prompt;
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input, [&] { return cancellation_token && cancellation_token->cancelled(); });
                if (!success){
                    json error_response = {{"error", "Max length reached"}};
                    send_response(error_response);
                    this->auto_chat_engine->clear_context();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            try {
                auto_chat_engine->generate(meta_info, length_limit, ostream, [&] { return cancellation_token && cancellation_token->cancelled(); });
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            std::string response_text = ss.str();
            auto history = this->auto_chat_engine->get_history();
            json response = {
                {"model", model},
                {"response", response_text},
                {"context", history.second},
                {"done", true},
                {"prompt_eval_count", meta_info.prompt_tokens},
                {"eval_count", meta_info.generated_tokens},
                {"total_duration", meta_info.total_duration},
                {"load_duration", meta_info.load_duration},
                {"prompt_eval_duration", meta_info.prefill_duration},
                {"eval_duration", meta_info.decoding_duration},
                {"done_reason", stop_reason_to_string(meta_info.stop_reason)}
            };
            // std::cout << "history: " << history.first << std::endl;
            send_response(response);
        }
    } catch (const std::exception& e) {
        json error_response = ExceptionResponse(e);
        send_response(error_response);
    }
}

///@brief Handle the chat request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_chat(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response,
                             std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        nlohmann::ordered_json messages = request["messages"];
        bool stream = request.value("stream", false);
        std::string model = request.value("model", current_model_tag);
        const std::string request_backend = request.value("backend", std::string());
        json options = request.value("options", json::object());
        int length_limit = options.value("num_predict", 4096);

        auto load_start_time = time_utils::now();
        if (!ensure_model_loaded(model, request_backend)) {
            json error_response = {{"error", "Failed to load " + model + " model!"}};
            send_response(error_response);
            return;
        }
        auto load_end_time = time_utils::now();
       
        configure_chat_engine_parameters(options, request);

        // messages = normalize_messages(messages);
        
        chat_meta_info_t meta_info;
        lm_uniform_input_t uniformed_input;
        uniformed_input.requested_max_new_tokens =
            normalize_requested_max_new_tokens(
                options.contains("num_predict")
                    ? std::optional<int>(options.at("num_predict").get<int>())
                    : std::nullopt);
        meta_info.load_duration = (uint64_t)time_utils::duration_ns(load_start_time, load_end_time).first;
        meta_info.max_prefill_len = this->prefill_chunk_len;
        header_print("FLM", "Start generating...");
        if (stream) {
            // Streaming response using streaming_ostream
            auto total_start_time = time_utils::now();
            streaming_ostream ostream(model, send_streaming_response, true);  // true for chat format
            uniformed_input.messages = messages;
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input, [&] { return cancellation_token && cancellation_token->cancelled(); });
                if (!success){
                    json error_response = {{"error", "Max length reached"}};
                    send_response(error_response);
                    this->auto_chat_engine->clear_context();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            try {
                auto_chat_engine->generate(meta_info, length_limit, ostream,
                    [&] { return cancellation_token && cancellation_token->cancelled(); });
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            auto total_end_time = time_utils::now();
            meta_info.total_duration = (uint64_t)time_utils::duration_ns(total_start_time, total_end_time).first;
            
            ostream.finalize_chat(meta_info);
            // auto history = this->chat_engine->get_history();
            // std::cout << "history: " << history.first << std::endl;
            this->auto_chat_engine->clear_context();
        } else {
            // Non-streaming response
            uniformed_input.messages = messages;
            auto total_start_time = time_utils::now();
            nullstream nstream;
            //std::string response_text = auto_chat_engine->generate_with_prompt(meta_info, uniformed_input, length_limit, std::cout);
            std::string response_text;
            try {
                response_text = auto_chat_engine->generate_with_prompt(
                    meta_info, uniformed_input, length_limit, nstream,
                    [&] { return cancellation_token && cancellation_token->cancelled(); });
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            //std::string response_text = chat_engine->generate_with_prompt(meta_info, prompts, length_limit, std::cout, payload);
            auto total_end_time = time_utils::now();
            meta_info.total_duration = (uint64_t)time_utils::duration_ns(total_start_time, total_end_time).first;
            
            json response = {
                {"model", model},
                {"message", {
                    {"role", "assistant"},
                    {"content", response_text},
                    {"images", nullptr}
                }},
                {"done", true},
                {"prompt_eval_count", meta_info.prompt_tokens},
                {"eval_count", meta_info.generated_tokens},
                {"total_duration", meta_info.total_duration},
                {"load_duration", meta_info.load_duration},
                {"prompt_eval_duration", meta_info.prefill_duration},
                {"eval_duration", meta_info.decoding_duration},
                {"done_reason", stop_reason_to_string(meta_info.stop_reason)}
            };
            send_response(response);
            
            // auto history = this->chat_engine->get_history();
            // std::cout << "history: " << history.first << std::endl;
            this->auto_chat_engine->clear_context();
        }
    } catch (const std::exception& e) {
        json error_response = ExceptionResponse(e);
        send_response(error_response);
    }
}

///@brief Handle the embeddings request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_embeddings(const json& request,
                                   std::function<void(const json&)> send_response,
                                   StreamResponseCallback send_streaming_response) {
    try {
        std::string model = request["model"];
        std::vector<std::string> inputs;

        if (request["input"].is_string()) {
            inputs.push_back(request["input"].get<std::string>());
        }
        else if (request["input"].is_array()) {
            for (const auto& item : request["input"]) {
                inputs.push_back(item.get<std::string>());
            }
        }

        json response;
        if (this->embed) {
            json embedding_data = json::array();
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::cout << "Embedding input[" << i << "]: " << "\n" << inputs[i] << std::endl;
                std::vector<float> embedding_result = this->auto_embedding_engine->embed(inputs[i], embedding_task_type_t::task_query);
                embedding_data.push_back({
                    {"object", "embedding"},
                    {"embedding", embedding_result},
                    {"index", i}
                });
            }
#else
            throw std::runtime_error("Embedding models are not supported in this build");
#endif

            response = {
                {"object", "list"},
                {"data", embedding_data},
                {"model", model},
                {"usage", {
                    {"prompt_tokens", 0},
                    {"total_tokens", 0}
                }}
            };
        }
        else {
            header_print("Warning", "No embedding model loaded");
        }
        send_response(response);
    }
    catch (const std::exception& e) {
        json error_response = ExceptionResponse(e);
        send_response(error_response);
    }
}

///@brief Handle the models request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_models(const json& request,
                               std::function<void(const json&)> send_response,
                               StreamResponseCallback send_streaming_response) {
    try {
        json models = supported_models.get_all_models_ollama();
        send_response(models);
    } catch (const std::exception& e) {
        json error_response = ExceptionResponse(e);
        send_response(error_response);
    }
}

///@brief Handle the version request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_version(const json& request,
                                std::function<void(const json&)> send_response,
                                StreamResponseCallback send_streaming_response) {
    std::string version = __FLM_VERSION__;
    
    json response = {{"version", version}};
    send_response(response);
}

///@brief Handle the models request (open ai)
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_models_openai(const json& request,
    std::function<void(const json&)> send_response,
    StreamResponseCallback send_streaming_response) {
    try {
        json models = supported_models.get_all_models_openai();
        send_response(models);
    }
    catch (const std::exception& e) {
        json error_response = { {"error", e.what()} };
        send_response(error_response);
    }
}

///@brief Handle the ps request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_ps(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    try {
        // Generate expires_at timestamp (1 hour from now)
        auto now = std::chrono::system_clock::now();
        auto expires_time = now + std::chrono::hours(1);
        auto expires_time_t = std::chrono::system_clock::to_time_t(expires_time);
        auto expires_tp = std::chrono::system_clock::from_time_t(expires_time_t);
        auto fractional_seconds = std::chrono::duration_cast<std::chrono::microseconds>(expires_time - expires_tp).count();
        
        // Get local time and timezone offset
        std::tm* local_tm = std::localtime(&expires_time_t);
        std::tm* utc_tm = std::gmtime(&expires_time_t);
        
        // Calculate timezone offset in minutes
        int offset_minutes = (local_tm->tm_hour - utc_tm->tm_hour) * 60 + (local_tm->tm_min - utc_tm->tm_min);
        if (local_tm->tm_mday != utc_tm->tm_mday) {
            offset_minutes += (local_tm->tm_mday > utc_tm->tm_mday) ? 1440 : -1440;
        }
        
        std::stringstream expires_ss;
        expires_ss.imbue(std::locale::classic()); // Use C locale to avoid commas
        expires_ss << std::put_time(local_tm, "%Y-%m-%dT%H:%M:%S");
        expires_ss << "." << std::setfill('0') << std::setw(5) << (fractional_seconds / 40); // 5 decimal places
        
        // Format timezone offset
        int offset_hours = offset_minutes / 60;
        int offset_mins = abs(offset_minutes % 60);
        expires_ss << (offset_minutes >= 0 ? "+" : "-") 
                  << std::setfill('0') << std::setw(2) << abs(offset_hours)
                  << ":" << std::setfill('0') << std::setw(2) << offset_mins;
        
        std::string expires_at = expires_ss.str();
        
        auto [new_current_model_tag, model_info] = supported_models.get_model_info(current_model_tag);
        json response = {
            {"models", json::array({
                {
                    {"name", current_model_tag},
                    {"model", current_model_tag},
                    {"size", model_info["size"]},
                    {"details", model_info["details"]},
                    {"expires_at", expires_at},
                }
            })}
        };
        // std::cout << "response: " << response.dump(4) << std::endl;
        send_response(response);
    } catch (const std::exception& e) {
        json error_response = ExceptionResponse(e);
        send_response(error_response);
    }
}

///@brief Handle the pull request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_pull(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Pull operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the push request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_push(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Push operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the delete request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_delete(const json& request,
                               std::function<void(const json&)> send_response,
                               StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Delete operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the copy request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_copy(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Copy operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the create request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_create(const json& request,
                               std::function<void(const json&)> send_response,
                               StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Create operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the openai chat completion request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
///@note This function only translates between the OpenAI wire format and
///      flm::request::ChatSession, which runs the request.
void RestHandler::handle_openai_chat_completion(const json& request,
                                               std::function<void(const json&)> send_response,
                                               StreamResponseCallback send_streaming_response,
                                               std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        // Extract OpenAI-style parameters
        flm::request::ChatRequest chat;
        chat.messages = request["messages"];
        std::string model = value_or(request, "model", current_model_tag);
        const std::string request_backend = value_or(request, "backend", std::string());
        bool stream = value_or(request, "stream", false);
        chat.max_new_tokens = openai_chat_budget(request);
        json tools = value_or(request, "tools", json::array());

        // tool_choice, parallel_tool_calls and response_format become a decoding constraint
        chat_constraint_request constraint_request;
        std::string request_error;
        if (!chat_constraint_request_from_openai(request, tools, constraint_request, request_error)) {
            send_response({{"error", {{"message", request_error}, {"type", "invalid_request_error"}, {"code", 400}}}});
            return;
        }
        if (constraint_request.tool_choice == chat_tool_choice::NONE) {
            // the model is not shown the tools at all: banning the call opener alone
            // leaves it writing the call's JSON as plain text
            tools = json::array();
        }

        auto load_start_time = time_utils::now();
        if (!ensure_model_loaded(model, request_backend)) {
            json error_response = {{"error", "Failed to load " + model + " model!"}};
            send_response(error_response);
            return;
        }
        auto load_end_time = time_utils::now();

        chat.model = model;
        chat.tools = tools;
        chat.constraint = constraint_request;
        chat.parameters = parse_engine_parameters(request);
        chat.load_duration_ns = (uint64_t)time_utils::duration_ns(load_start_time, load_end_time).first;
        chat.max_prefill_len = this->prefill_chunk_len;
        chat.stream = stream;
        auto is_cancelled = [&] { return cancellation_token->cancelled(); };

        const json single_turn_error = {{"error", "This model only supports single-turn requests. Multi-turn conversation history (assistant messages) is not allowed."}};
        const json context_error = {
            {"error", {
            {"message", "Max length reached!"},
            {"type", "model_error"},
            {"code", 400}
            }}
        };

        if (stream){
            // Create a wrapper callback that passes the pre-formatted SSE string directly
            cancellation_token->reset();
            bool stream_started = false;
            auto openai_stream_callback = [&send_streaming_response, &stream_started](const std::string& data, bool is_final) {
                stream_started = true;
                json data_json = data;
                send_streaming_response(data_json, is_final);
                };
            chat_completion_sse sink(model, openai_stream_callback, stream_include_usage(request));  // streaming in chat completion format
            try {
                flm::request::ChatOutcome outcome = chat_session.run_chat(*auto_chat_engine, chat, &sink, is_cancelled);
                switch (outcome.end) {
                case flm::request::ChatEnd::single_turn_rejected:
                    send_response(single_turn_error);
                    return;
                case flm::request::ChatEnd::context_exceeded:
                    send_response(context_error);
                    return;
                default:
                    // completed, cancelled or cancelled during prefill: the client still gets its finish chunk
                    sink.finalize(outcome.meta, outcome);
                    break;
                }
            } catch (const flm::request::InvalidRequest& e) {
                // the request asks for something this model's format cannot do, or a bad schema
                send_response({{"error", {{"message", e.what()}, {"type", "invalid_request_error"}, {"code", 400}}}});
                return;
            } catch (const flm::request::RequestSetupError&) {
                throw;
            } catch (const ModelRequestError& error) {
                const json error_response = ModelErrorResponse(error);
                if (stream_started) {
                    send_streaming_response(json("data: " + error_response.dump() + "\n\n"), false);
                    send_streaming_response(json("data: [DONE]\n\n"), true);
                } else {
                    send_response(error_response);
                }
                return;
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                return;
            }
        }
        else {
            flm::request::ChatOutcome outcome;
            try {
                outcome = chat_session.run_chat(*auto_chat_engine, chat, nullptr, is_cancelled);
            } catch (const flm::request::InvalidRequest& e) {
                send_response({{"error", {{"message", e.what()}, {"type", "invalid_request_error"}, {"code", 400}}}});
                return;
            } catch (const flm::request::RequestSetupError&) {
                throw;
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                return;
            }
            switch (outcome.end) {
            case flm::request::ChatEnd::single_turn_rejected:
                send_response(single_turn_error);
                return;
            case flm::request::ChatEnd::context_exceeded:
                send_response(context_error);
                return;
            case flm::request::ChatEnd::prefill_cancelled:
                send_response(json());
                return;
            default:
                break;
            }

            const chat_meta_info_t& meta_info = outcome.meta;
            json choices = build_chat_choices(outcome.message, meta_info);
            json response = {
                {"id", "fastflowlm-chat-completion"},
                {"object", "chat.completion"},
                {"created", static_cast<long long>(std::time(nullptr))},
                {"model", model},
                {"choices", choices},
                {"usage", {
                    {"prompt_tokens", meta_info.prompt_tokens},
                    {"prompt_tokens_details", {{"cached_tokens", meta_info.cached_prompt_tokens}}},
                    {"completion_tokens", meta_info.generated_tokens},
                    {"total_tokens", meta_info.prompt_tokens + meta_info.generated_tokens},
                    {"kv_token_occupancy_rate_percentage", (float)outcome.active_kv_tokens / (float)outcome.max_kv_tokens * 100},
                    {"load_duration", static_cast<double>(meta_info.load_duration) / 1'000'000'000},
                    {"prefill_duration_ttft", static_cast<double>(meta_info.prefill_duration) / 1'000'000'000},
                    {"decoding_duration", static_cast<double>(meta_info.decoding_duration) / 1'000'000'000},
                    {"prefill_speed_tps", static_cast<double>(meta_info.prompt_tokens - meta_info.cached_prompt_tokens) / static_cast<double>(meta_info.prefill_duration) * 1'000'000'000},
                    {"decoding_speed_tps", static_cast<double>(meta_info.generated_tokens) / static_cast<double>(meta_info.decoding_duration) * 1'000'000'000},
                }},
                {"service_tier", "default"}
            };
            send_response(response);
        }

    } catch (const std::exception& e) {
        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "server_error"},
                {"code", 500}
            }}
        };
        send_response(error_response);
    }
}

///@brief Handle the openai audio transcriptions request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_openai_audio_transcriptions(const json& request,
                                        std::function<void(const json&)> send_response,
                                        StreamResponseCallback send_streaming_response,
                                        std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        std::string model = request["model"];
        std::string file_content = request["file"].get<std::string>();
        std::vector<uint8_t> audio_raw(file_content.begin(), file_content.end());
        std::string language = request.value("language", "");
        std::string prompt = request.value("prompt", "");
        std::string response_format = request.value("response_format", "json");
        bool stream = request.value("stream", false);
        json response;
        if (response_format != "json" && response_format != "text" && response_format != "verbose_json" &&
            response_format != "srt" && response_format != "vtt") {
            throw std::invalid_argument("unsupported response_format: " + response_format +
                                        " (json, text, verbose_json, srt or vtt)");
        }
        // Segments need the timestamps; json and text are the same transcript without them.
        const bool timed = response_format == "verbose_json" || response_format == "srt" || response_format == "vtt";
        if (this->asr) {
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
            this->whisper_engine->load_audio(audio_raw);
            header_print("FLM", "Transforming audio to text...");
            // Transcripts are user speech: log them only when FLM_LOG_RAW_OUTPUT asks.
            const char* log_env = std::getenv("FLM_LOG_RAW_OUTPUT");
            const bool log_transcript = log_env != nullptr && log_env[0] != '\0' && std::strcmp(log_env, "0") != 0;
            std::ostream no_log(nullptr);
            std::ostream& transcript_log = log_transcript ? std::cout : no_log;
            transcript_log << "Audio content: " << std::flush;
            std::pair<std::string, std::string> audio_result = this->whisper_engine->generate(Whisper::whisper_task_type_t::e_transcribe, true, timed, transcript_log, language, prompt);
            std::string audio_context = audio_result.first;
            std::string language_name = audio_result.second;
            double duration = this->whisper_engine->audio_duration();
            transcript_log << std::endl;
#else
            throw std::runtime_error("ASR models are not supported in this build");
            std::string audio_context;
            std::string language_name;
            double duration = 0.0;
#endif

            if (response_format == "json") {
                response = {
                    {"model", model},
                    {"text", audio_context}
                };
            }
            else if (response_format == "text") {
                response = audio_context;  // a string goes out as a text/plain body
            }
            else {
                std::vector<transcript::Segment> segments = transcript::split_segments(audio_context, duration);
                if (response_format == "srt") {
                    response = transcript::to_srt(segments);
                }
                else if (response_format == "vtt") {
                    response = transcript::to_vtt(segments);
                }
                else {
                    // Whisper's decoder keeps no per-segment statistics: tokens,
                    // avg_logprob, compression_ratio and no_speech_prob are
                    // placeholders that no client filter drops a segment on.
                    json segments_json = json::array();
                    std::string text;
                    for (const transcript::Segment& segment : segments) {
                        segments_json.push_back({
                            {"id", (int)segments_json.size()},
                            {"seek", 0},
                            {"start", segment.start},
                            {"end", segment.end},
                            {"text", segment.text},
                            {"tokens", json::array()},
                            {"temperature", 0.0},
                            {"avg_logprob", 0.0},
                            {"compression_ratio", 0.0},
                            {"no_speech_prob", 0.0}
                        });
                        text += segment.text;
                    }
                    // OpenAI names the language in lower case, as openai-whisper's LANGUAGES.
                    std::transform(language_name.begin(), language_name.end(), language_name.begin(),
                                   [](unsigned char c) { return (char)std::tolower(c); });
                    response = {
                        {"task", "transcribe"},
                        {"language", language_name},
                        {"duration", duration},
                        {"text", text},
                        {"segments", segments_json},
                        {"model", model}
                    };
                }
            }
        }
        else {
            header_print("Warning", "No asr model loaded, cannot load audio file");
        }
        send_response(response);
        //this->whisper_engine->clear_context();
    }
    catch (const std::invalid_argument& e) {
        // A bad language or response_format. A 400, as OpenAI's: clients retry a 500.
        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "invalid_request_error"},
                {"code", 400}
            }}
        };
        send_response(error_response);
    }
    catch (const std::exception& e) {
        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "server_error"},
                {"code", 500}
            }}
        };
        send_response(error_response);
    }
}

///@brief Handle the openai completion request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_openai_completion(const json& request,
    std::function<void(const json&)> send_response,
    StreamResponseCallback send_streaming_response,
    std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        // Extract OpenAI-style parameters
        std::string prompt = request["prompt"];
        std::string model = request.value("model", current_model_tag);
        const std::string request_backend = request.value("backend", std::string());
        bool stream = request.value("stream", false);

        // direct return if model not supported
        if (!supported_models.is_model_supported(model)) {
            throw std::runtime_error("Model " + model + " is not supported.");
        }

        const std::optional<int> max_new_tokens = request.contains("max_tokens")
            ? std::optional<int>(request.at("max_tokens").get<int>())
            : std::nullopt;

        if (!ensure_model_loaded(model, request_backend)) {
            json error_response = {{"error", "Failed to load " + model + " model!"}};
            send_response(error_response);
            return;
        }

        flm::request::CompletionRequest completion;
        completion.prompt = prompt;
        completion.max_new_tokens = max_new_tokens;
        completion.parameters = parse_engine_parameters(request);
        completion.max_prefill_len = this->prefill_chunk_len;
        completion.stream = stream;
        auto is_cancelled = [&] { return cancellation_token && cancellation_token->cancelled(); };

        if (stream) {
            // Create a wrapper callback that passes the pre-formatted SSE string directly
            bool stream_started = false;
            auto openai_stream_callback = [&send_streaming_response, &stream_started](const std::string& data, bool is_final) {
                stream_started = true;
                json data_json = data;
                send_streaming_response(data_json, is_final);
                };
            streaming_ostream_openai ostream(model, openai_stream_callback, stream_include_usage(request));  // streaming in completion format
            try {
                flm::request::CompletionOutcome outcome =
                    chat_session.run_completion(*auto_chat_engine, completion, ostream, is_cancelled);
                if (outcome.context_exceeded) {
                    json error_response = { {"error", "Max length reached"} };
                    send_response(error_response);
                    return;
                }
                ostream.finalize(outcome.meta);
            } catch (const ModelRequestError& error) {
                const json error_response = ModelErrorResponse(error);
                if (stream_started) {
                    send_streaming_response(json("data: " + error_response.dump() + "\n\n"), false);
                    send_streaming_response(json("data: [DONE]\n\n"), true);
                } else {
                    send_response(error_response);
                }
                return;
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                return;
            }
        }
        else {
            std::stringstream ss;
            wstream_buf obuf(ss);
            std::ostream ostream(&obuf);
            flm::request::CompletionOutcome outcome;
            try {
                outcome = chat_session.run_completion(*auto_chat_engine, completion, ostream, is_cancelled);
                if (outcome.context_exceeded) {
                    json error_response = { {"error", "Max length reached"} };
                    send_response(error_response);
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = ExceptionResponse(e);
                send_response(error_response);
                return;
            }
            const chat_meta_info_t& meta_info = outcome.meta;
            std::string response_text = ss.str();

            json response = {
                {"id", "fastflowlm-chat-completion"},
                {"object", "text_completion"},
                {"created", (int)std::time(nullptr)},
                {"model", model},
                {"choices", json::array({
                    {
                        {"text", response_text},
                        {"index", 0},
                        {"logprobs", nullptr},
                        {"finish_reason", stop_reason_to_string(meta_info.stop_reason)}
                    }
                })},
                {"usage", {
                    {"prompt_tokens", meta_info.prompt_tokens},
                    {"prompt_tokens_details", {{"cached_tokens", meta_info.cached_prompt_tokens}}},
                    {"completion_tokens", meta_info.generated_tokens},
                    {"total_tokens", meta_info.prompt_tokens + meta_info.generated_tokens}
                }}
            };
            send_response(response);
        }
    }
    catch (const std::exception& e) {
        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "server_error"},
                {"code", 500}
            }}
        };
        send_response(error_response);
    }
}
