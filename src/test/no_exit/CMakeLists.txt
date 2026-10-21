# Load-path failures that were exit(1) before the throw conversion. No NPU, no engine
# library and no device is touched; the constructors and loaders only read files.
#   test_no_exit          model_list, LM_Config, Whisper_Config and (optionally) Tokenizer
#   test_no_exit_automodel AutoModel::_shared_setup_tokenizer and _shared_initialize_legacy_npu
cmake_minimum_required(VERSION 3.22)
project(no_exit_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

# lm_config.hpp, utils/utils.hpp and AutoModel/automodel.hpp include the XRT device
# header, so the XRT and Boost include directories are needed even though no device is
# used. The same directories build-flm.ps1 passes to the main build:
#   -DXRT_INCLUDE_DIR=<.../runtime_src/core/include>  -DXRT_GEN_INCLUDE_DIR=<dir with xrt/detail/version.h>
#   -DFLM_VCPKG_INCLUDE_DIR=<vcpkg installed x64-windows include>   (Boost)
# Optional. Enables the Tokenizer case and the AutoModel target (both link tokenizers-cpp):
#   -DFLM_TOKENIZERS_LIB_DIR="<dir with tokenizers_cpp.lib and tokenizers_c.lib>;<dir with sentencepiece.lib>"
#   -DFLM_TOKENIZERS_INCLUDE_DIR=<third_party/tokenizers-cpp/include>
#   -DFLM_XRT_LIB_DIR=<dir with xrt_coreutil.lib>
set(XRT_INCLUDE_DIR "C:/dev/XRT/src/runtime_src/core/include" CACHE PATH "Where XRT headers live")
set(XRT_GEN_INCLUDE_DIR "" CACHE PATH "Extra include dir for generated XRT headers (xrt/detail/version.h)")
set(FLM_VCPKG_INCLUDE_DIR "" CACHE PATH "vcpkg installed include dir (Boost for XRT)")
set(FLM_TOKENIZERS_LIB_DIR "" CACHE PATH "Directory with the built tokenizers-cpp libraries")
set(FLM_TOKENIZERS_INCLUDE_DIR "${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp/include" CACHE PATH "tokenizers-cpp headers")
set(FLM_XRT_LIB_DIR "" CACHE PATH "Directory with xrt_coreutil.lib")

set(NO_EXIT_DEFINITIONS
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
    _CRT_NONSTDC_NO_DEPRECATE
    __WINDOWS__
    DISABLE_ABI_CHECK=1
    CMAKE_INSTALL_PREFIX="${CMAKE_CURRENT_BINARY_DIR}/install-prefix"
    CMAKE_XCLBIN_PREFIX="${CMAKE_CURRENT_BINARY_DIR}/xclbin-prefix")

# Shared include and link settings for both executables.
function(no_exit_configure TARGET_NAME)
    target_compile_definitions(${TARGET_NAME} PRIVATE ${NO_EXIT_DEFINITIONS})
    target_include_directories(${TARGET_NAME} PRIVATE
        "${FLM_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_LIST_DIR}"
        "${XRT_INCLUDE_DIR}")
    if(XRT_GEN_INCLUDE_DIR)
        target_include_directories(${TARGET_NAME} PRIVATE "${XRT_GEN_INCLUDE_DIR}")
    endif()
    if(FLM_VCPKG_INCLUDE_DIR)
        target_include_directories(${TARGET_NAME} PRIVATE "${FLM_VCPKG_INCLUDE_DIR}")
    endif()
    if(WIN32)
        # SHGetFolderPathA in utils.cpp
        target_link_libraries(${TARGET_NAME} PRIVATE shell32)
    endif()
endfunction()

# ---- model_list, LM_Config, Whisper_Config, Tokenizer -------------------------------

set(NO_EXIT_SOURCES
    test_no_exit.cpp
    "${FLM_SOURCE_DIR}/common/utils.cpp")
set(NO_EXIT_TOKENIZER_DEFINITION "")
if(FLM_TOKENIZERS_LIB_DIR)
    list(APPEND NO_EXIT_SOURCES "${FLM_SOURCE_DIR}/common/tokenizer/tokenizer.cpp")
    set(NO_EXIT_TOKENIZER_DEFINITION FLM_TEST_TOKENIZER=1)
endif()

add_executable(test_no_exit ${NO_EXIT_SOURCES})
no_exit_configure(test_no_exit)
if(FLM_TOKENIZERS_LIB_DIR)
    target_compile_definitions(test_no_exit PRIVATE ${NO_EXIT_TOKENIZER_DEFINITION})
    target_include_directories(test_no_exit PRIVATE "${FLM_TOKENIZERS_INCLUDE_DIR}")
    target_link_directories(test_no_exit PRIVATE "${FLM_TOKENIZERS_LIB_DIR}")
    target_link_libraries(test_no_exit PRIVATE tokenizers_cpp tokenizers_c sentencepiece
        ntdll ws2_32 Bcrypt userenv)
endif()

# ---- AutoModel load steps ----------------------------------------------------------

if(FLM_TOKENIZERS_LIB_DIR)
    # automodel.cpp refers to the tokenizer and sampler, the registry in model_backend.cpp, and
    # the chat library (with the xgrammar static library it needs).
    include("${FLM_SOURCE_DIR}/common/chat/chat_sources.cmake")
    add_executable(test_no_exit_automodel
        test_no_exit_automodel.cpp
        "${FLM_SOURCE_DIR}/common/AutoModel/automodel.cpp"
        "${FLM_SOURCE_DIR}/common/AutoModel/model_backend.cpp"
        "${FLM_SOURCE_DIR}/common/modules/sampler.cpp"
        "${FLM_SOURCE_DIR}/common/tokenizer/tokenizer.cpp"
        "${FLM_SOURCE_DIR}/common/utils.cpp"
        ${FLM_CHAT_SOURCES})
    no_exit_configure(test_no_exit_automodel)
    target_include_directories(test_no_exit_automodel PRIVATE
        "${FLM_SOURCE_DIR}/common" "${FLM_SOURCE_DIR}/../include")
    target_link_libraries(test_no_exit_automodel PRIVATE flm_xgrammar)
    target_compile_definitions(test_no_exit_automodel PRIVATE ${NO_EXIT_TOKENIZER_DEFINITION})
    target_include_directories(test_no_exit_automodel PRIVATE "${FLM_TOKENIZERS_INCLUDE_DIR}")
    target_link_directories(test_no_exit_automodel PRIVATE "${FLM_TOKENIZERS_LIB_DIR}")
    if(FLM_XRT_LIB_DIR)
        target_link_directories(test_no_exit_automodel PRIVATE "${FLM_XRT_LIB_DIR}")
    endif()
    target_link_libraries(test_no_exit_automodel PRIVATE tokenizers_cpp tokenizers_c sentencepiece
        xrt_coreutil ntdll ws2_32 Bcrypt userenv psapi iphlpapi crypt32 secur32 advapi32 normaliz wldap32)
endif()

enable_testing()
add_test(NAME test_no_exit COMMAND test_no_exit)
if(FLM_TOKENIZERS_LIB_DIR)
    add_test(NAME test_no_exit_automodel COMMAND test_no_exit_automodel)
endif()
