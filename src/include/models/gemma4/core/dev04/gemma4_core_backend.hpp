/// \file gemma4_core_backend.hpp
/// \brief The ryzenai-corelib backend for Gemma 4 (E2B and E4B)
/// \note Compiled only when FLM_ENABLE_CORE is on.
#pragma once

#include "AutoModel/model_backend.hpp"

#include <cstdint>
#include <stdexcept>

namespace flm::gemma4 {

/// \brief rows in one page of the paged KV cache, and the longest unpaged context
inline constexpr std::uint32_t kPageSize = 4096;

/// \brief the longest context validated on hardware
/// \note corelib pages Gemma 4 up to the model's own 131072 and refuses past a
///       GGUF's smaller gemma4.context_length; 16384 is what was validated.
inline constexpr std::uint32_t kCoreContextLimit = 16384;

/// \brief one below the longest context: the last position has to be readable
inline constexpr std::uint32_t kCoreDecodeLimit = kCoreContextLimit - 1;

/// \return true when `length` is 1..4096, or a multiple of kPageSize up to kCoreContextLimit
inline bool ContextFits(std::uint32_t length) {
    if (length < 1 || length > kCoreContextLimit) return false;
    return length <= kPageSize || length % kPageSize == 0;
}

/// \note Inline: AutoModel reads it while assembling the BackendContext,
///       before any engine exists.
inline flm::backend::BackendTraits core_traits() {
    flm::backend::BackendTraits traits;
    traits.needs_npu_xclbin = false;
    traits.supports_preemption = false;
    traits.max_context_length = kCoreContextLimit;
    traits.validate_load = [](const flm::backend::BackendContext& context) {
        if (!ContextFits(context.context_length))
            throw std::out_of_range("Gemma 4 core context length must be in "
                                    "1..4096, or a multiple of 4096 up to 16384");
    };
    return traits;
}

flm::backend::BackendFactory core_factory();

}  // namespace flm::gemma4
